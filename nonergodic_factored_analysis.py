"""
Non-Ergodic Mess3: Factored Representation Analysis  (N components)
=====================================================================

Constructs a non-ergodic training dataset from N Mess3 HMMs with different
parameters (x and alpha). Each training sequence is generated entirely by one ergodic
component
=====================================================================

"""

import os
import sys
import warnings
from itertools import combinations

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import torch.nn.functional as F
# ── path fix: saccade-ms/datasets.py shadows the HuggingFace datasets package ──, needs to be commented out at the end
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if "/home/vikash/Notebooks/saccade-ms" not in p]
import datasets          # loads from site-packages, cached in sys.modules
sys.path = _saved_path   # restore so other saccade imports keep working
from transformer_lens import HookedTransformer, HookedTransformerConfig

warnings.filterwarnings("ignore")



SAVE_DIR = "plots_nonergodic"
os.makedirs(SAVE_DIR, exist_ok=True)

# Set to a .pt path to skip training and load a saved checkpoint instead.
CHECKPOINT_PATH: str | None = "plots_nonergodic/checkpoint.pt"

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

DEVICE  = "cuda:3" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 100

# ── N Mess3 process configurations ───────────────────────────────────────────
# Choose parameters that produce visually distinct fractals:
#   x     ∈ (0, 0.5)
#   alpha ∈ [0, 1]:   emission fidelity (higher = cleaner fractal)
PROCESS_CONFIGS = [
    {"x": 0.04, "alpha": 0.85},   # process A
    {"x": 0.08, "alpha": 0.60},   # process B  
    {"x": 0.12, "alpha": 0.8},   # process C 
]
N = len(PROCESS_CONFIGS)

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN_BATCH    = 64
NUM_STEPS      = 50_001
SNAPSHOT_EVERY = 500

# ── Analysis ─────────────────────────────────────────────────────────────────
ANALYSIS_BATCH = 4_000   # total sequences per analysis pass (split equally across components)
K_SUBSPACE     = 2    # PCA dims per factor (each Mess3 belief is 2-D)

# ── Theoretical dim predictions ────────────────────────────────────
#   Factored: K dims per component  → K * N
#   Joint:    N*3-state simplex, 1 normalization constraint → N*3 - 1
FACTORED_DIM = K_SUBSPACE * N
JOINT_DIM    = N * 3 - 1

# ── Per-component colors and labels (consistent across all plots) ─────────────
COMP_COLORS  = plt.cm.tab10(np.linspace(0, 0.9, max(N, 1)))
COMP_LABELS  = [chr(65 + i) for i in range(N)]   # ["A", "B", "C", ...]


# ════════════════════════════════════════════════════════════════════════════════
# CLASSES
# ════════════════════════════════════════════════════════════════════════════════

class Mess3Process:
    """
    Single ergodic Mess3 HMM: 3 hidden states, 3 tokens.
    Parameterised by x (transition spread) and alpha (emission fidelity).
    """

    def __init__(self, x: float = 0.05, alpha: float = 0.85, device: str = DEVICE):
        self.x      = x
        self.alpha  = alpha
        self.device = device
        self.T      = self._build_T()           # (3, 3, 3)
        self.prior  = torch.ones(3, device=device) / 3

    def _build_T(self):
        x, alpha = self.x, self.alpha
        T = torch.zeros(3, 3, 3)
        for s in range(3):
            for t in range(3):
                for tok in range(3):
                    p_trans = 1 - 2 * x if s == t else x
                    p_emit  = alpha if tok == t else (1 - alpha) / 2
                    T[s, t, tok] = p_trans * p_emit
        return T.to(self.device)

    def find_next_state(self, cur: torch.Tensor):
        """cur: (B,) CPU int tensor. Returns (next_states, tokens) on CPU."""
        B     = cur.shape[0]
        probs = self.T[cur.to(self.device)].view(B, -1)
        idx   = torch.multinomial(probs, 1).squeeze(1)
        return (idx // 3).cpu(), (idx % 3).cpu()

    def generate_sequences(self, length: int, batch_size: int):
        seqs, states = [], []
        cur = torch.randint(0, 3, (batch_size,))
        for _ in range(length):
            nxt, tok = self.find_next_state(cur)
            seqs.append(tok); states.append(nxt); cur = nxt
        return torch.stack(seqs, 1), torch.stack(states, 1)

    def find_belief_states(self, sequences: torch.Tensor):
        """Exact Bayesian belief updates → (B, L+1, 3)."""
        B, L = sequences.shape
        b = self.prior.unsqueeze(0).expand(B, -1).clone()
        out = [b]
        for t in range(L):
            T_tok = self.T.permute(2, 0, 1)[sequences[:, t].to(self.device)]
            b = torch.bmm(b.unsqueeze(1), T_tok).squeeze(1)
            b = b / b.sum(1, keepdim=True)
            out.append(b)
        return torch.stack(out, 1)

    def find_belief_loss(self, sequences: torch.Tensor, include_start: bool = True):
        B, L = sequences.shape
        bs   = self.find_belief_states(sequences)
        Tm   = self.T.sum(-2)
        pred = torch.bmm(bs[:, :-1], Tm.expand(B, -1, -1))
        tp   = pred.gather(2, sequences.to(self.device).unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(tp.clamp(1e-10))
        return loss[:, (0 if include_start else 1):]


class NonErgodicMess3:
    """
    Non-ergodic mixture of N Mess3 processes.

    Every sequence is generated ENTIRELY by one component (label ∈ {0,..,N-1}).
    No within-sequence switching occurs. The transformer sees only tokens
    from {0, 1, 2} and must infer which component is active from context alone.

    A combined (N*3)-state block-diagonal HMM supports exact Bayesian belief
    computation for ground-truth comparisons.
    """

    def __init__(self, processes: list, mix_prob: list | None = None,
                 device: str = DEVICE):
        self.processes = processes
        self.N         = len(processes)
        self.device    = device
        # Equal mixing by default
        if mix_prob is None:
            mix_prob = [1.0 / self.N] * self.N
        assert len(mix_prob) == self.N and abs(sum(mix_prob) - 1) < 1e-6
        self.mix_prob   = mix_prob
        self.T_combined = self._build_T_combined()   # (N*3, N*3, 3)
        self.prior      = self._build_prior()        # (N*3,)

    def _build_T_combined(self):
        S = self.N * 3
        T = torch.zeros(S, S, 3, device=self.device)
        for i, proc in enumerate(self.processes):
            T[i*3:(i+1)*3, i*3:(i+1)*3, :] = proc.T
        return T

    def _build_prior(self):
        p = torch.zeros(self.N * 3, device=self.device)
        for i, w in enumerate(self.mix_prob):
            p[i*3:(i+1)*3] = w / 3
        return p

    def generate_sequences(self, length: int, batch_size: int):
        """
        Returns
        -------
        sequences : (B, L) int tensor, tokens in {0, 1, 2}
        labels    : (B,)   int tensor, component index in {0, …, N-1}
        """
        labels = torch.from_numpy(
            np.random.choice(self.N, size=batch_size, p=self.mix_prob)
        ).long()
        cur  = torch.randint(0, 3, (batch_size,))
        seqs = []
        for _ in range(length):
            nxt = torch.zeros(batch_size, dtype=torch.long)
            tok = torch.zeros(batch_size, dtype=torch.long)
            for i, proc in enumerate(self.processes):
                mask = (labels == i)
                if mask.any():
                    ns, t = proc.find_next_state(cur[mask])
                    nxt[mask] = ns; tok[mask] = t
            seqs.append(tok); cur = nxt
        return torch.stack(seqs, 1), labels

    def find_belief_states_combined(self, sequences: torch.Tensor):
        """Full (N*3)-state Bayesian belief → (B, L+1, N*3)."""
        B, L = sequences.shape
        b = self.prior.unsqueeze(0).expand(B, -1).clone()
        out = [b]
        for t in range(L):
            tok   = sequences[:, t].to(self.device)
            T_tok = self.T_combined.permute(2, 0, 1)[tok]   # (B, N*3, N*3)
            b     = torch.bmm(b.unsqueeze(1), T_tok).squeeze(1)
            b     = b / b.sum(1, keepdim=True)
            out.append(b)
        return torch.stack(out, 1)                           # (B, L+1, N*3)

    def find_belief_loss(self, sequences: torch.Tensor, include_start: bool = True):
        """NTP cross-entropy under the optimal (N*3)-state predictor."""
        B, L   = sequences.shape
        bs     = self.find_belief_states_combined(sequences)
        T_emit = self.T_combined.sum(1)                      # (N*3, 3)
        pred   = torch.bmm(bs[:, :-1], T_emit.expand(B, -1, -1))   # (B, L, 3)
        tp     = pred.gather(2, sequences.to(self.device).unsqueeze(-1)).squeeze(-1)
        loss   = -torch.log(tp.clamp(1e-10))
        return loss[:, (0 if include_start else 1):]


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS UTILITIES  
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_activations(model: HookedTransformer,
                    sequences: torch.Tensor,
                    flatten: bool = True,
                    batch_size: int = 256) -> np.ndarray:
    """
    Run model on sequences[:, :-1]; return final residual stream activations.
    flatten=True  → (B*(L-1), d_model)
    flatten=False → (B, L-1, d_model)

    Processes in mini-batches of batch_size to avoid OOM from run_with_cache
    storing all intermediate activations for a large batch at once.
    """
    chunks = []
    for start in range(0, len(sequences), batch_size):
        inp = sequences[start:start + batch_size, :-1].to(DEVICE)
        _, cache = model.run_with_cache(inp)
        chunk = cache["ln_final.hook_normalized"].detach().cpu().numpy()
        chunks.append(chunk)
        del cache          # free GPU activation cache immediately
    acts = np.concatenate(chunks, axis=0)
    return acts.reshape(-1, acts.shape[-1]) if flatten else acts


def compute_cev(acts: np.ndarray, max_k: int = 60) -> np.ndarray:
    """Cumulative explained-variance ratio for up to max_k PCA components."""
    n = min(max_k, acts.shape[0] - 1, acts.shape[1])
    return np.cumsum(PCA(n_components=n).fit(acts).explained_variance_ratio_)


def effective_dim(acts: np.ndarray, threshold: float = 0.95, max_k: int = 60) -> int:
    """Smallest k such that the top-k PCs cover ≥ threshold of variance."""
    cev = compute_cev(acts, max_k=max_k)
    return min(int(np.searchsorted(cev, threshold)) + 1, len(cev))


def get_subspace(acts: np.ndarray, k: int) -> np.ndarray:
    """Mean-centre acts, PCA, return top-k components as (k, d_model)."""
    return PCA(n_components=k).fit(acts - acts.mean(0)).components_


def subspace_overlap(V1: np.ndarray, V2: np.ndarray) -> float:
    """
    Normalised Frobenius overlap between two k-dim subspaces.
    V1, V2 : (k, d) orthonormal row matrices.
    Returns scalar in [0, 1]; 0 = perfectly orthogonal, 1 = identical.
    """
    M = V1 @ V2.T
    return float(np.sum(M ** 2) / V1.shape[0])


def recover_geometry(acts: np.ndarray, targets: np.ndarray):
    """Linear regression acts → targets. Returns (predictions, R²)."""
    reg  = LinearRegression().fit(acts, targets)
    pred = reg.predict(acts)
    ss_res = ((targets - pred) ** 2).sum()
    ss_tot = ((targets - targets.mean(0)) ** 2).sum()
    return pred, float(1.0 - ss_res / (ss_tot + 1e-12))


def normalise_beliefs(b: np.ndarray) -> np.ndarray:
    """Row-normalise a (N, 3) array of unnormalised belief marginals."""
    return b / b.sum(1, keepdims=True).clip(1e-8)


# ════════════════════════════════════════════════════════════════════════════════
# SIMPLEX HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def to_cartesian(b: np.ndarray):
    return b[:, 1] + 0.5 * b[:, 2], (np.sqrt(3) / 2) * b[:, 2]


def draw_triangle(ax, vertex_labels=("A", "B", "C"), color="k"):
    vs = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(vs[:, 0], vs[:, 1], c=color, lw=1.5)
    offsets = [(-0.07, -0.07), (0.01, -0.07), (-0.04, 0.03)]
    for v, lbl, off in zip(vs[:3], vertex_labels, offsets):
        ax.text(v[0] + off[0], v[1] + off[1], lbl, fontsize=11)


def scatter_simplex(ax, beliefs: np.ndarray, title: str = "",
                    s: float = 3, alpha: float = 0.5):
    xs, ys = to_cartesian(beliefs.clip(0, 1))
    ax.scatter(xs, ys, c=beliefs.clip(0, 1), s=s, alpha=alpha, marker=".")
    draw_triangle(ax)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 – THEORETICAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def section1_theory(processes: list, nonergodic: NonErgodicMess3,
                    n_points: int = 300_000):
    """
    Plots:
      (a) One 2-simplex geometry per Mess3 component
      (b) Component-separation curve: p(component i | context) vs. position
          for sequences drawn from component i — should rise from 1/N to ~1
    """
    print(f"\n── SECTION 1: Theoretical belief state geometry  (N={N}) ─────────")

    n_seqs = max(n_points // SEQ_LEN, 500)
    theory_beliefs = []   # list of N np arrays, each (n_seqs*(L), 3)

    for i, proc in enumerate(processes):
        print(f"  Sampling component-{i} belief states  "
              f"(x={proc.x}, α={proc.alpha}) …")
        seqs, _ = proc.generate_sequences(SEQ_LEN, n_seqs)
        bs = proc.find_belief_states(seqs)[:, 1:].reshape(-1, 3).cpu().numpy()
        theory_beliefs.append(bs)

    # Component-separation curves
    print("  Computing component-separation curves …")
    n_sep   = 2_000
    seqs_mix, labels = nonergodic.generate_sequences(SEQ_LEN, n_sep)
    bs_full = nonergodic.find_belief_states_combined(seqs_mix)   # (B, L+1, N*3)
    pos     = np.arange(SEQ_LEN + 1)

    # ── Layout: (2 rows) top = N simplex plots, bottom = separation curve ─────
    n_cols   = N
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(4 * n_cols, 8),
                             gridspec_kw={"height_ratios": [1, 1]})
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f"Figure 1 – Theoretical Analysis: Non-Ergodic Mess3 ({N} components)",
                 fontsize=12)

    for i, (proc, bs) in enumerate(zip(processes, theory_beliefs)):
        scatter_simplex(axes[0, i], bs[::5],
                        title=(f"Component {COMP_LABELS[i]}\n"
                               f"x={proc.x},  α={proc.alpha}"),
                        s=2, alpha=0.35)

    # Separation curves — one panel per component
    for i in range(N):
        ax       = axes[1, i]
        comp_mask = (labels == i).numpy()
        # p(component i | context) for sequences that ARE from component i
        p_i = bs_full[:, :, i*3:(i+1)*3].sum(-1).cpu().numpy()   # (B, L+1)

        mean_self = p_i[comp_mask].mean(0)
        std_self  = p_i[comp_mask].std(0)
        ax.plot(pos, mean_self, color=COMP_COLORS[i], lw=2,
                label=f"comp-{COMP_LABELS[i]} seqs")
        ax.fill_between(pos,
                        mean_self - std_self,
                        mean_self + std_self,
                        alpha=0.15, color=COMP_COLORS[i])
        # Show the mean for sequences from OTHER components (should decrease)
        other_mask = (labels != i).numpy()
        if other_mask.any():
            ax.plot(pos, p_i[other_mask].mean(0),
                    color="gray", lw=1, ls="--", alpha=0.6, label="other seqs")
        ax.axhline(1.0 / N, color="black", ls=":", lw=1,
                   label=f"prior 1/{N}")
        ax.set_xlabel("Context position", fontsize=9)
        ax.set_ylabel(f"p(comp-{COMP_LABELS[i]} | ctx)", fontsize=9)
        ax.set_title(f"Separation: component {COMP_LABELS[i]}", fontsize=9)
        ax.legend(fontsize=7); ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    path = f"{SAVE_DIR}/01_theory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {path}")

    return theory_beliefs


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 – TRAINING
# ════════════════════════════════════════════════════════════════════════════════

def build_model():
    # Scale d_model slightly with N so there is capacity for N*2 subspaces
    d_model = max(64, 16 * N)
    cfg = HookedTransformerConfig(
        d_model=d_model, d_head=16,
        n_layers=3, n_ctx=SEQ_LEN, n_heads=max(4, d_model // 16),
        d_mlp=d_model * 4, d_vocab=3,
        device=DEVICE, act_fn="relu",
    )
    return HookedTransformer(cfg), cfg


def section2_train(nonergodic: NonErgodicMess3):
    """
    Trains the transformer and collects metrics every SNAPSHOT_EVERY steps.

    Returns
    -------
    model            : trained HookedTransformer
    history          : dict of per-snapshot metrics
    optimal_loss     : float – theoretical lower bound (N*3-state predictor)
    analysis_seqs    : fixed analysis batch (ANALYSIS_BATCH sequences)
    analysis_labels  : component label for each analysis sequence
    """
    print(f"\n── SECTION 2: Training  (N={N}) ──────────────────────────────────")

    model, cfg = build_model()
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Fixed validation set
    val_seqs, _  = nonergodic.generate_sequences(SEQ_LEN, 2_000)
    val_in       = val_seqs[:, :-1].to(DEVICE)
    val_tgt      = val_seqs[:, 1:].to(DEVICE)

    # Theoretical lower bound
    with torch.no_grad():
        optimal_loss = nonergodic.find_belief_loss(
            val_seqs, include_start=False
        ).mean().item()
    print(f"  Optimal ({N*3}-state belief) loss : {optimal_loss:.4f}")

    # Fixed analysis batch split by component
    analysis_seqs, analysis_labels = nonergodic.generate_sequences(
        SEQ_LEN, ANALYSIS_BATCH
    )
    component_seqs = {}
    for i in range(N):
        mask = (analysis_labels == i)
        component_seqs[i] = analysis_seqs[mask]
        print(f"  Component {i}: {mask.sum():4d} analysis sequences")

    n_pairs = N * (N - 1) // 2
    pair_indices = list(combinations(range(N), 2))

    history: dict = {
        "step":           [],
        "train_loss":     [],
        "val_loss":       [],
        "eff_dim_mixed":  [],
        "eff_dim_comp":   [],   # list[N floats] per snapshot
        "overlap_pairs":  [],   # list[n_pairs floats] per snapshot (at K_SUBSPACE)
        "avg_overlap":    [],
        "cev":            [],
        "overlap_vs_k":   [],   # list of (K_max, n_pairs) arrays per snapshot
    }

    print(f"  Training for {NUM_STEPS:,} steps (snapshot every {SNAPSHOT_EVERY}) …")
    for step in tqdm(range(NUM_STEPS)):
        seqs, _ = nonergodic.generate_sequences(SEQ_LEN, TRAIN_BATCH)
        logits  = model(seqs[:, :-1].to(DEVICE))
        loss    = F.cross_entropy(
            logits.reshape(-1, cfg.d_vocab),
            seqs[:, 1:].to(DEVICE).reshape(-1),
        )
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if step % SNAPSHOT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                v_loss = F.cross_entropy(
                    model(val_in).reshape(-1, cfg.d_vocab),
                    val_tgt.reshape(-1),
                ).item()

            acts_mix  = get_activations(model, analysis_seqs)
            acts_comp = {}
            for i in range(N):
                if len(component_seqs[i]) >= K_SUBSPACE + 1:
                    acts_comp[i] = get_activations(model, component_seqs[i])

            # Per-component effective dim
            ed_comp = [
                effective_dim(acts_comp[i]) if i in acts_comp else 0
                for i in range(N)
            ]

            # Pairwise subspace overlaps at K_SUBSPACE (for backward compat)
            subspaces = {
                i: get_subspace(acts_comp[i], K_SUBSPACE)
                for i in acts_comp
            }
            overlaps = [
                subspace_overlap(subspaces[i], subspaces[j])
                for i, j in pair_indices
                if i in subspaces and j in subspaces
            ]
            avg_ov = float(np.mean(overlaps)) if overlaps else float("nan")

            # Overlap vs k: for each k=1..K_max compute mean pairwise overlap
            K_max = min(20, min(
                a.shape[0] - 1 for a in acts_comp.values()
            ) if acts_comp else K_SUBSPACE)
            ov_vs_k = []   # (K_max, n_pairs)
            for k in range(1, K_max + 1):
                subs_k = {i: get_subspace(acts_comp[i], k) for i in acts_comp}
                ov_k   = [
                    subspace_overlap(subs_k[i], subs_k[j])
                    for i, j in pair_indices
                    if i in subs_k and j in subs_k
                ]
                ov_vs_k.append(ov_k if ov_k else [float("nan")] * max(len(pair_indices), 1))
            ov_vs_k = np.array(ov_vs_k)   # (K_max, n_pairs)

            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(v_loss)
            history["eff_dim_mixed"].append(effective_dim(acts_mix))
            history["eff_dim_comp"].append(ed_comp)
            history["overlap_pairs"].append(overlaps)
            history["avg_overlap"].append(avg_ov)
            history["cev"].append(compute_cev(acts_mix, max_k=60))
            history["overlap_vs_k"].append(ov_vs_k)

            torch.cuda.empty_cache()   # release any fragmented cache after snapshot
            model.train()

    print("  Training complete.")

    # ── Loss curve ─────────────────────────────────────────────────────────────
    steps = np.array(history["step"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, history["train_loss"], "b-",  alpha=0.6, lw=1.5, label="Train loss")
    ax.plot(steps, history["val_loss"],   "r-",  lw=2,             label="Val loss")
    ax.axhline(optimal_loss, color="k", ls="--", lw=1.5,
               label=f"Optimal ({N*3}-state belief) = {optimal_loss:.3f}")
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss (nats)")
    ax.set_title(f"Section 2 – Training and Validation Loss  (N={N} components)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{SAVE_DIR}/02_loss_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {path}")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = f"{SAVE_DIR}/checkpoint.pt"
    torch.save({
        "model_state":    model.state_dict(),
        "history":        history,
        "optimal_loss":   optimal_loss,
        "analysis_seqs":  analysis_seqs,
        "analysis_labels": analysis_labels,
    }, ckpt_path)
    print(f"  ✓ Checkpoint saved → {ckpt_path}")

    return model, history, optimal_loss, analysis_seqs, analysis_labels


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3a – FACTOR GEOMETRY RECOVERY
# ════════════════════════════════════════════════════════════════════════════════

def section3a_factor_recovery(model, nonergodic: NonErgodicMess3,
                               analysis_seqs, analysis_labels,
                               theory_beliefs: list):
    """

    Layout: 2 rows × N columns
      Top row    : ground-truth belief geometry for each component
      Bottom row : geometry recovered from the residual stream via
                   linear regression (using component-i sequences only)
    """
    print(f"\n── SECTION 3a: Factor geometry recovery  (N={N}) ─────────────────")

    # Align ground-truth beliefs to model positions (beliefs 1..L-1 match acts)
    beliefs_full = (
        nonergodic.find_belief_states_combined(analysis_seqs)[:, 1:-1, :]
        .cpu().numpy()
    )                                                     # (B, L-1, N*3)
    Lm1 = analysis_seqs.shape[1] - 1

    fig, axes = plt.subplots(2, N, figsize=(4 * N, 9))
    if N == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        f"Figure 2 – Factor Geometry Recovery  (N={N})\n"
        "Top: ground truth   Bottom: recovered from residual stream",
        fontsize=11,
    )

    for i in range(N):
        comp_mask = (analysis_labels == i).numpy()
        if not comp_mask.any():
            axes[0, i].axis("off"); axes[1, i].axis("off"); continue

        # Activations and beliefs for component-i sequences
        acts_i = get_activations(model, analysis_seqs[comp_mask])
        gt_i   = normalise_beliefs(
            beliefs_full[comp_mask, :, i*3:(i+1)*3].reshape(-1, 3)
        )
        pred_i, r2_i = recover_geometry(acts_i, gt_i)
        pred_i = normalise_beliefs(pred_i.clip(0))

        print(f"  R² component {i}: {r2_i:.4f}")

        # Ground-truth column
        scatter_simplex(
            axes[0, i], theory_beliefs[i][::5],
            title=(f"GT – component {COMP_LABELS[i]}\n"
                   f"x={nonergodic.processes[i].x}, "
                   f"α={nonergodic.processes[i].alpha}"),
            s=2, alpha=0.3,
        )
        # Recovered column
        scatter_simplex(
            axes[1, i], pred_i,
            title=f"Recovered – component {COMP_LABELS[i]}\nR² = {r2_i:.3f}",
            s=3, alpha=0.4,
        )

    fig.tight_layout()
    path = f"{SAVE_DIR}/03a_factor_recovery.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3b/c – CEV AND EFFECTIVE DIMENSIONALITY
# ════════════════════════════════════════════════════════════════════════════════

def section3bc_cev_effdim(history: dict):
    """

    (b) CEV curves coloured by training step, with dashed lines at the
        factored (K*N) and joint (N*3-1) theoretical predictions.
    (c) Effective dimensionality (95 % threshold) over training.
    """
    print("\n── SECTION 3b/c: CEV and effective dimensionality ────────────────")

    steps         = np.array(history["step"])
    eff_dim_mixed = np.array(history["eff_dim_mixed"])
    n_snap        = len(history["cev"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Figure 3 – CEV and Effective Dimensionality  (N={N})\n"
        f"Factored pred = {FACTORED_DIM}D   |   Joint pred = {JOINT_DIM}D",
        fontsize=11,
    )

    # ── (b) CEV curves ────────────────────────────────────────────────────────
    ax   = axes[0]
    cmap = plt.cm.plasma(np.linspace(0.05, 0.95, n_snap))
    for i, (cev, step) in enumerate(zip(history["cev"], steps)):
        label = f"step {step:,}" if i % max(1, n_snap // 6) == 0 else ""
        ax.plot(np.arange(1, len(cev) + 1), cev,
                color=cmap[i], alpha=0.75, lw=1.2, label=label)
    ax.axvline(FACTORED_DIM, color="green",  ls="--", lw=2,
               label=f"Factored ({FACTORED_DIM}D)")
    ax.axvline(JOINT_DIM,    color="orange", ls="--", lw=2,
               label=f"Joint ({JOINT_DIM}D)")
    ax.axhline(0.95, color="gray", ls=":", lw=1, label="95 % threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("(a) CEV curves over training", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0.5, max(JOINT_DIM * 2, 12))

    # ── (c) Effective dimensionality ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, eff_dim_mixed, "b-o", ms=5, lw=2,
            label="Mixed-batch effective dim")
    ax.axhline(FACTORED_DIM, color="green",  ls="--", lw=2,
               label=f"Factored ({FACTORED_DIM}D)")
    ax.axhline(JOINT_DIM,    color="orange", ls="--", lw=2,
               label=f"Joint ({JOINT_DIM}D)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dims for 95 % variance")
    ax.set_title("(b) Effective dimensionality over training", fontsize=10)
    ax.legend()

    fig.tight_layout()
    path = f"{SAVE_DIR}/03bc_cev_effdim.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3d/e – VARY-ONE AND SUBSPACE ORTHOGONALITY
# ════════════════════════════════════════════════════════════════════════════════

def section3de_varyone_orthogonality(history: dict):
    """

    (1) Vary-one effective dimensionality per component over training.
        Each component's "vary-one" set = sequences from that component only.
        Should converge to K_SUBSPACE = 2 per factor (one simplex = 2-D).

    (2) All N*(N-1)/2 pairwise subspace overlaps (thin lines) plus their
        mean (thick line) over training.  Should approach 0 (orthogonal)
        if the model learns factored representations.
    """
    print("\n── SECTION 3d/e: Vary-one analysis and subspace orthogonality ────")

    steps      = np.array(history["step"])
    # eff_dim_comp: list of snapshots, each a list of N values
    ed_comp_arr = np.array(history["eff_dim_comp"])           # (n_snap, N)
    pair_labels = [f"comp {COMP_LABELS[i]}–{COMP_LABELS[j]}" for i, j in combinations(range(N), 2)]
    n_pairs     = len(pair_labels)
    # overlap_pairs: list of snapshots, each a list of n_pairs values
    overlap_arr = np.array(history["overlap_pairs"])          # (n_snap, n_pairs)
    avg_overlap = np.array(history["avg_overlap"])            # (n_snap,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Figure 4 – Vary-One Subspace Analysis and Orthogonality  (N={N})",
        fontsize=11,
    )

    # ── (d) Vary-one effective dimensionality ─────────────────────────────────
    ax = axes[0]
    for i in range(N):
        ax.plot(steps, ed_comp_arr[:, i],
                color=COMP_COLORS[i], marker="o", ms=4, lw=2,
                label=f"Component {COMP_LABELS[i]} (x={nonergodic_ref.processes[i].x}, "
                      f"α={nonergodic_ref.processes[i].alpha})")
    ax.axhline(K_SUBSPACE, color="green", ls="--", lw=2,
               label=f"Factored pred ({K_SUBSPACE}D per component)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dims for 95 % variance (vary-one)")
    ax.set_title("(a) Vary-one effective dimensionality per component", fontsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)

    # ── (e) Pairwise subspace overlap ─────────────────────────────────────────
    ax     = axes[1]
    n_snap = len(steps)

    if "overlap_vs_k" in history and history["overlap_vs_k"]:
        # New style: x = components k, y = overlap (log), one line per snapshot
        snap_cmap  = plt.cm.viridis(np.linspace(0.05, 0.95, n_snap))
        label_idxs = sorted({0, n_snap // 4, n_snap // 2, n_snap - 1})

        for snap_idx, (step, ov_vs_k) in enumerate(
                zip(steps, history["overlap_vs_k"])):
            ov_arr  = np.array(ov_vs_k)           # (K_max, n_pairs)
            mean_ov = np.nanmean(ov_arr, axis=1)  # (K_max,)
            ks      = np.arange(1, len(mean_ov) + 1)
            labeled = snap_idx in label_idxs
            lw      = 0.8 if labeled else 0.4
            alpha   = 0.9 if labeled else 0.35
            label   = f"Step {step:,}" if labeled else ""
            for p in range(ov_arr.shape[1]):
                ax.plot(ks, np.clip(ov_arr[:, p], 1e-3, 1.0),
                        color=snap_cmap[snap_idx], lw=0.3, alpha=0.12)
            ax.plot(ks, np.clip(mean_ov, 1e-3, 1.0),
                    color=snap_cmap[snap_idx], lw=lw, alpha=alpha, label=label)

        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e0)
        ax.set_xlabel("Components (k)", fontsize=10)
        ax.set_ylabel("Subspace Overlap", fontsize=10)
        ax.set_title(
            f"(b) Pairwise subspace overlap vs. subspace size  ({n_pairs} pairs)",
            fontsize=10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlim(0, 11)
    else:
        # Fallback (old history without overlap_vs_k): overlap vs training step
        print("  [section3de] 'overlap_vs_k' not in history — "
              "re-run training to get the k-axis plot. Showing step-axis fallback.")
        pair_cmap = plt.cm.Set2(np.linspace(0, 0.9, max(n_pairs, 1)))
        for k, lbl in enumerate(pair_labels):
            if overlap_arr.shape[1] > k:
                ax.plot(steps, overlap_arr[:, k],
                        color=pair_cmap[k], lw=1.2, alpha=0.65, label=lbl)
        ax.plot(steps, avg_overlap, color="black", lw=2.5, label="Mean")
        ax.set_xlabel("Training step"); ax.set_ylabel("Normalised subspace overlap")
        ax.set_title(f"(b) Pairwise subspace orthogonality  ({n_pairs} pairs)",
                     fontsize=10)
        ax.set_ylim(-0.05, 1.05)

    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    path = f"{SAVE_DIR}/03de_varyone_orthogonality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 – META-BELIEF SUBSPACE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def section4_meta_belief(model: HookedTransformer,
                         analysis_seqs:   torch.Tensor,
                         analysis_labels: torch.Tensor,
                         late_start: int = 60,
                         n_traj:     int = 1):
    """
    Mechanistic interpretability pipeline for the "meta-belief" subspace V_meta:
    the 2-D plane that encodes which component (A / B / C / …) is active.

    Step 1 – Centroid method
        Collect activations from late tokens (position late_start…end), where
        the model is confident about the component.  Compute one centroid per
        component, then run PCA on the N centroids → 2-D V_meta.

    Step 2 – Orthogonality verification
        Measure Overlap(V_meta, V_i) = (1/2) Tr(P_meta P_i) for each local
        subspace V_i.  Should be ~0 if meta-belief is factored away from local
        belief geometry.

    Step 3 – Superposition-to-collapse trajectory
        Project full sequences (token 0 → end) onto V_meta.
        Expected geometry: start at centroid of triangle → random walk →
        rapid lock-in to one corner → frozen on V_meta while local
        belief continues to move in V_i.

    Returns V_meta (2, d_model), local_subspaces dict, overlaps dict.
    """
    print(f"\n── SECTION 4: Meta-belief subspace analysis  (N={N}) ─────────────")

    labels = analysis_labels.numpy()

    # ── Step 1a: position-aware activations ──────────────────────────────────
    acts_full = get_activations(model, analysis_seqs, flatten=False)
    # shape: (B, L-1, d_model)  — position t = model state after seeing t+1 tokens
    B, Lm1, d = acts_full.shape

    late_idx = late_start - 1          # first index in [0, Lm1) for "late" tokens
    late_idx = max(0, min(late_idx, Lm1 - 1))

    # ── Step 1b: class centroids from late tokens ─────────────────────────────
    centroids = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        mask_i   = (labels == i)
        late_i   = acts_full[mask_i, late_idx:, :]   # (n_i, late_len, d)
        centroids[i] = late_i.reshape(-1, d).mean(0)

    # ── Step 1c: PCA on the N centroids → V_meta ─────────────────────────────
    k_meta       = 2    # always use exactly 2 dims so Overlap(V_meta, V_i) is (1/2)Tr(P_meta P_i)
    centroid_mean = centroids.mean(0)
    pca_meta     = PCA(n_components=k_meta)
    pca_meta.fit(centroids - centroid_mean)
    V_meta = pca_meta.components_        # (k_meta, d) orthonormal

    print(f"  V_meta shape          : {V_meta.shape}")
    print(f"  Explained variance    : "
          f"{pca_meta.explained_variance_ratio_.round(4)}")
    print(f"  (100 % expected — N centroids span exactly an (N-1)-D affine plane)")

    # ── Step 2: local subspaces + orthogonality check ────────────────────────
    local_subspaces = {}
    for i in range(N):
        mask_i = (labels == i)
        if mask_i.sum() > K_SUBSPACE:
            acts_i = acts_full[mask_i].reshape(-1, d)
            local_subspaces[i] = get_subspace(acts_i, K_SUBSPACE)   # (2, d)

    print("\n  Overlap(V_meta, V_i)   [0=orthogonal, 1=identical]:")
    overlaps_meta = {}
    for i, V_i in local_subspaces.items():
        # Overlap = (1/k_meta) * ||V_meta @ V_i.T||_F^2  =  subspace_overlap
        ov = subspace_overlap(V_meta, V_i)
        overlaps_meta[i] = ov
        tag = "orthogonal ✓" if ov < 0.1 else ("weak overlap" if ov < 0.3 else "NOT orthogonal ✗")
        print(f"    Overlap(V_meta, V_{COMP_LABELS[i]}): {ov:.4f}  [{tag}]")

    # ── Step 3: superposition-to-collapse trajectories ───────────────────────
    # Select n_traj sequences per component
    traj_seqs, traj_labs = [], []
    for i in range(N):
        idx_i = np.where(labels == i)[0][:n_traj]
        traj_seqs.append(analysis_seqs[idx_i])
        traj_labs.extend([i] * len(idx_i))
    traj_seqs = torch.cat(traj_seqs, dim=0)
    traj_labs = np.array(traj_labs)

    traj_acts = get_activations(model, traj_seqs, flatten=False)   # (n*N, L-1, d)
    # Project onto V_meta
    traj_2d = (traj_acts - centroid_mean) @ V_meta.T               # (n*N, L-1, 2)
    corner_2d = (centroids - centroid_mean) @ V_meta.T             # (N, 2)

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Figure 5 – Meta-Belief Subspace  V_meta  (N={N} components)\n"
        f"Step 1: centroid PCA  |  Step 2: orthogonality  |  Step 3: collapse trajectories",
        fontsize=11,
    )

    # ── Panel 1: Late-token cloud projected onto V_meta ───────────────────────
    ax = axes[0]
    late_acts_all   = acts_full[:, late_idx:, :].reshape(-1, d)
    late_labs_all   = np.repeat(labels, Lm1 - late_idx)
    proj_late       = (late_acts_all - centroid_mean) @ V_meta.T    # (n_total, 2)

    for i in range(N):
        mi = (late_labs_all == i)
        ax.scatter(proj_late[mi, 0], proj_late[mi, 1],
                   c=[COMP_COLORS[i]], s=4, alpha=0.25, rasterized=True)
        ax.scatter(*corner_2d[i], c=[COMP_COLORS[i]], s=250, marker="*",
                   zorder=6, edgecolors="k", linewidths=0.8,
                   label=f"μ_{COMP_LABELS[i]}  (comp {COMP_LABELS[i]})")
        ax.annotate(f"μ_{COMP_LABELS[i]}", corner_2d[i], fontsize=10,
                    xytext=(0, 10), textcoords="offset points", ha="center")

    ax.set_xlabel("V_meta PC1")
    ax.set_ylabel("V_meta PC2")
    ax.set_title(f"Late-token activations on V_meta\n(tokens {late_start}–{SEQ_LEN})",
                 fontsize=9)
    ax.legend(fontsize=8, markerscale=0.8)
    ax.set_aspect("equal")

    # ── Panel 2: Orthogonality bar chart ──────────────────────────────────────
    ax = axes[1]
    bar_labels = [f"V_{COMP_LABELS[i]}" for i in range(N)]
    bar_vals   = [overlaps_meta.get(i, float("nan")) for i in range(N)]
    bars = ax.bar(bar_labels, bar_vals,
                  color=[COMP_COLORS[i] for i in range(N)],
                  alpha=0.8, edgecolor="k", linewidth=0.8)
    ax.axhline(0.1, color="green",  ls="--", lw=1.5, label="Threshold 0.1")
    ax.axhline(1.0, color="tomato", ls="--", lw=1.5, label="Identical (1.0)")
    ax.axhline(0.0, color="green",  ls="-",  lw=0.8, alpha=0.4)
    for bar, val in zip(bars, bar_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.015, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Overlap(V_meta, V_i)")   # generic label intentional
    ax.set_title("Step 2: Orthogonality of V_meta vs local subspaces\n"
                 "(→ 0 means meta-belief is cleanly factored)", fontsize=9)
    ax.legend(fontsize=8)

    # ── Panel 3: Superposition-to-collapse trajectories ───────────────────────
    ax = axes[2]
    # Draw target triangle
    tri_pts = np.vstack([corner_2d, corner_2d[:1]])
    ax.plot(tri_pts[:, 0], tri_pts[:, 1], "k--", lw=1.2, alpha=0.5, zorder=1)
    for i in range(N):
        ax.scatter(*corner_2d[i], c=[COMP_COLORS[i]], s=250, marker="*",
                   zorder=6, edgecolors="k", linewidths=0.8)
        ax.annotate(f"μ_{COMP_LABELS[i]}", corner_2d[i], fontsize=10,
                    xytext=(0, 10), textcoords="offset points", ha="center")
    # Prior centre
    centre_2d = corner_2d.mean(0)
    ax.scatter(*centre_2d, c="black", s=120, marker="o", zorder=7,
               label="Prior centre (token 0)")

    # Trajectories — fade from light (early) to dark (late)
    for seq_i in range(len(traj_labs)):
        comp  = traj_labs[seq_i]
        traj  = traj_2d[seq_i]           # (L-1, 2)
        n_pos = traj.shape[0]
        alphas = np.linspace(0.08, 0.85, n_pos - 1)
        for t in range(n_pos - 1):
            ax.plot(traj[t:t+2, 0], traj[t:t+2, 1],
                    color=COMP_COLORS[comp], lw=0.9, alpha=float(alphas[t]))
        # Token-0 dot
        ax.scatter(*traj[0], c="black", s=18, marker="o", alpha=0.6, zorder=5)

    ax.set_xlabel("V_meta PC1")
    ax.set_ylabel("V_meta PC2")
    ax.set_title("Step 3: Superposition → Collapse trajectories\n"
                 "(colour = true component,  fade = token position)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = f"{SAVE_DIR}/04_meta_belief.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {path}")

    return V_meta, local_subspaces, overlaps_meta


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

# Module-level reference so section3de can label axes with process params
nonergodic_ref: NonErgodicMess3 | None = None


def main():
    global nonergodic_ref

    print("=" * 70)
    print(f"Non-Ergodic Mess3: Factored Representation Analysis  (N={N})")
    print("=" * 70)
    print(f"  Device        : {DEVICE}")
    for i, cfg in enumerate(PROCESS_CONFIGS):
        print(f"  Process {i}     : x={cfg['x']},  α={cfg['alpha']}")
    print(f"  Training      : {NUM_STEPS:,} steps,  batch={TRAIN_BATCH}")
    print(f"  Theory preds  : factored = {FACTORED_DIM}D,  joint = {JOINT_DIM}D")
    print(f"  Outputs       : {SAVE_DIR}/")
    print("=" * 70)

    # Build processes and non-ergodic dataset
    processes    = [Mess3Process(**cfg) for cfg in PROCESS_CONFIGS]
    nonergodic   = NonErgodicMess3(processes)
    nonergodic_ref = nonergodic     # expose to section3de for labels

    # ── Section 1: Theory ─────────────────────────────────────────────────────
    theory_beliefs = section1_theory(processes, nonergodic)

    # ── Section 2: Training or checkpoint loading ─────────────────────────────
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"\n── Loading checkpoint: {CHECKPOINT_PATH} ──────────────────────")
        ckpt           = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model, _       = build_model()
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)
        history        = ckpt["history"]
        optimal_loss   = ckpt["optimal_loss"]
        analysis_seqs   = ckpt["analysis_seqs"].cpu()
        analysis_labels = ckpt["analysis_labels"].cpu()
        print(f"  ✓ Loaded  (steps recorded: {history['step'][-1]:,})")
    else:
        if CHECKPOINT_PATH:
            print(f"  [warn] CHECKPOINT_PATH='{CHECKPOINT_PATH}' not found — training from scratch.")
        model, history, optimal_loss, analysis_seqs, analysis_labels = (
            section2_train(nonergodic))

    # ── Section 3:  activation-geometry analysis ───────────────────────
    section3a_factor_recovery(
        model, nonergodic, analysis_seqs, analysis_labels, theory_beliefs
    )
    section3bc_cev_effdim(history)
    section3de_varyone_orthogonality(history)

    # ── Section 4: Meta-belief subspace ───────────────────────────────────────
    section4_meta_belief(model, analysis_seqs, analysis_labels)

    print("\n" + "=" * 70)
    print(f"All plots saved to  {SAVE_DIR}/")
    print("  01_theory.png                  – per-component geometries + separation")
    print("  02_loss_curves.png             – train / val vs. optimal bound")
    print("  03a_factor_recovery.png        – ground truth vs. recovered geometry")
    print("  03bc_cev_effdim.png            – CEV curves + effective dim")
    print("  03de_varyone_orthogonality.png – vary-one dims + pairwise overlaps")
    print("  04_meta_belief.png             – V_meta subspace: centroids, orthogonality, collapse")
    print("=" * 70)


if __name__ == "__main__":
    main()
