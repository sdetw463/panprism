"""
Dual-mode sequence-to-function (S2F) inference runners and feature masking utilities.

Modes:
  - `mit` (default open-source mode): Flashzoi-4x (524 kb, MIT license) and
    Pangolin-12x (GPL-3.0), with all AlphaGenome channels and peak coordinates masked.
  - `full`: adds AlphaGenome-450M (262 kb / 524 kb / 1 Mb, forward + reverse-complement,
    with diagonal-normalized physical 3D contacts) alongside Flashzoi-4x and Pangolin-12x.

Also provides FP32/BF16 numerical parity helpers and a PyTorch 2.x SDPA attention shim.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from panprism.breakpoint_graph import (
    HaplotypeGraphMapping,
    pool_alphagenome_on_gpu,
    pool_borzoi_on_gpu,
    reverse_complement,
)
from panprism.contact_physics import undo_squashed_scale

# ---------------------------------------------------------------------------
# Inference mode definitions and feature schemas
# ---------------------------------------------------------------------------

SUPPORTED_MODES: Tuple[str, ...] = ("mit", "full")
ALPHAGENOME_SUPPORTED_WINDOWS: Tuple[int, ...] = (131072, 262144, 524288, 1048576)
FLASHZOI_REPLICATE_REPOS: Tuple[str, ...] = tuple(
    f"johahi/flashzoi-replicate-{i}" for i in range(4)
)
PANGOLIN_ENSEMBLE_SIZE: int = 12


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a PanPrism inference mode."""

    name: str
    uses_alphagenome: bool
    backbones: Tuple[str, ...]
    default_window_len: int
    supported_windows: Tuple[int, ...]
    alphagenome_track_fraction: float
    alphagenome_coord_fraction: float
    description: str


MODE_CONFIGS: Dict[str, ModeConfig] = {
    "mit": ModeConfig(
        name="mit",
        uses_alphagenome=False,
        backbones=("flashzoi-4x", "pangolin-12x"),
        default_window_len=524288,
        supported_windows=(524288,),
        alphagenome_track_fraction=0.0,
        alphagenome_coord_fraction=0.0,
        description="Open-source default mode (MIT Flashzoi-4x 524kb + GPL-3.0 Pangolin-12x; no AlphaGenome)",
    ),
    "full": ModeConfig(
        name="full",
        uses_alphagenome=True,
        backbones=("alphagenome", "flashzoi-4x", "pangolin-12x"),
        default_window_len=524288,
        supported_windows=ALPHAGENOME_SUPPORTED_WINDOWS,
        alphagenome_track_fraction=1.0,
        alphagenome_coord_fraction=1.0,
        description="Full multi-backbone mode (AlphaGenome-524kb/1Mb + Flashzoi-4x + Pangolin-12x)",
    ),
}

# 29-channel diploid allelic expression feature schema
PANPRISM_V2_FEATURE_NAMES: Tuple[str, ...] = (
    "ag_exon_polya_dens",       # 0
    "ag_exon_polya_sum",        # 1
    "ag_exon_total_dens",       # 2
    "ag_exon_gtex_lcl_dens",    # 3
    "ag_exon_h3k36me3",         # 4
    "ag_exon_polr2a",           # 5
    "ag_prom_cage_s",           # 6
    "ag_prom_atac_dnase",       # 7
    "ag_prom_hist_act",         # 8
    "ag_prom_polr2a_ep300",     # 9
    "ag_prom_ctcf_yy1",         # 10
    "ag_ep_1d_abc",             # 11
    "ag_ep_3d_gm12878_abc",     # 12
    "ag_ep_3d_microc_abc",      # 13
    "ag_ep_3d_abc_frac_max",    # 14
    "ag_ep_3d_ctcf_loop",       # 15
    "ag_tss_3d_total_z",        # 16
    "ag_splice_canonical_prob", # 17
    "ag_splice_usage_lcl",      # 18
    "fz_exon_gm_rna_dens",      # 19 (MIT)
    "fz_exon_gtex_blood_dens",  # 20 (MIT)
    "fz_prom_cage_dnase",       # 21 (MIT)
    "fz_ep_1d_abc",             # 22 (MIT)
    "fz_ep_3d_gm_abc",          # 23 (AG 3D Hi-C coupled -> stripped in MIT!)
    "ag_prom_cage_anti",        # 24
    "sv_ins_prox_act",          # 25
    "sv_ins_genebody_interf",   # 26
    "sv_del_bp_ctcf_disrupt",   # 27
    "exons_surv_frac",          # 28 (MIT geometry)
)
MIT_ALLOWED_PANPRISM_V2_INDICES: Tuple[int, ...] = (19, 20, 21, 22, 28)
PB_V2_FEATURE_NAMES: Tuple[str, ...] = PANPRISM_V2_FEATURE_NAMES
MIT_ALLOWED_PB_V2_INDICES: Tuple[int, ...] = MIT_ALLOWED_PANPRISM_V2_INDICES

_AG_3D_TOKENS: Tuple[str, ...] = ("3d", "hic", "microc", "abc_frac", "ctcf_loop", "tad_insul")
_AG_DERIVED_PREFIXES: Tuple[str, ...] = (
    "ag_",
    "sv_ins_",
    "sv_del_bp_ctcf",
    "sv_flank_",
    "sv_span_",
    "sv_splice_",
    "exon_splice_",
    "sv_promoter_",
    "sv_abs_disruption",
    "ep_1d_ag",
    "ep_3d_",
)


def get_mode_config(mode: str = "mit") -> ModeConfig:
    """Return immutable `ModeConfig` for `'mit'` (default) or `'full'`."""
    m = str(mode).lower()
    if m not in MODE_CONFIGS:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")
    return MODE_CONFIGS[m]


def resolve_backbones_for_mode(
    mode: str = "mit",
    backbone: Optional[str] = None,
) -> List[str]:
    """
    Resolve active S2F backbones for a given `mode` (`'mit'` or `'full'`), strictly blocking
    AlphaGenome under `mode='mit'` (`0%` AlphaGenome guarantee).
    """
    cfg = get_mode_config(mode)
    if backbone is None or backbone == "auto":
        return list(cfg.backbones)
    bb = str(backbone).lower()
    valid_bbs = ("alphagenome", "borzoi", "flashzoi", "flashzoi-4x", "pangolin-12x")
    if bb not in valid_bbs:
        raise ValueError(f"Unsupported backbone {backbone!r}; expected one of {valid_bbs}")
    if not cfg.uses_alphagenome and bb == "alphagenome":
        raise ValueError(
            "Backbone 'alphagenome' cannot be used in mode='mit' (0% AlphaGenome). "
            "Pass mode='full' to enable AlphaGenome."
        )
    return [bb]


def build_v2_runner_spec(
    mode: str = "mit",
    backbone: str = "auto",
    window_len: Optional[int] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Build a validated, serializable runner specification (CPU-safe for
    `--dry-run`, telemetry, and pipeline orchestration).
    """
    cfg = get_mode_config(mode)
    backbones = resolve_backbones_for_mode(mode=cfg.name, backbone=backbone)
    uses_ag = "alphagenome" in backbones

    eff_win = int(window_len) if window_len is not None else cfg.default_window_len
    if uses_ag and eff_win not in ALPHAGENOME_SUPPORTED_WINDOWS:
        raise ValueError(
            f"Unsupported AlphaGenome window_len={eff_win}; expected one of {ALPHAGENOME_SUPPORTED_WINDOWS}"
        )
    if not uses_ag and any(b in ("borzoi", "flashzoi", "flashzoi-4x") for b in backbones) and eff_win != 524288:
        raise ValueError(
            f"Borzoi/Flashzoi backbones require window_len=524288, got {eff_win}"
        )

    return {
        "mode": cfg.name,
        "backbones": backbones,
        "uses_alphagenome": uses_ag,
        "alphagenome_track_fraction": 1.0 if uses_ag else 0.0,
        "alphagenome_coord_fraction": 1.0 if uses_ag else 0.0,
        "window_len": eff_win,
        "use_physical_3d": uses_ag,
        "use_rc_average": True,
        "use_unsquashed_flashzoi": any(b in ("flashzoi", "flashzoi-4x", "borzoi") for b in backbones),
        "device": device,
    }


def is_alphagenome_feature_name(name: str) -> bool:
    """
    Return `True` if a named feature depends on AlphaGenome 1D tracks, AlphaGenome 3D
    Hi-C/Micro-C contact maps, or AlphaGenome-derived SV chromatin nodes.
    """
    low = str(name).lower()
    if any(tok in low for tok in _AG_3D_TOKENS):
        return True
    if low.startswith("ag_") or "_ag_" in low or low.endswith("_ag"):
        return True
    if low.startswith(("fz_", "flashzoi_", "bz_", "borzoi_", "pang_", "pangolin_", "openspliceai_")):
        return False
    if any(low.startswith(pref) for pref in _AG_DERIVED_PREFIXES):
        return True
    return False


def mask_features_for_mode(
    features: np.ndarray,
    feature_names: Sequence[str],
    mode: str = "mit",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce mode-consistent feature channel masking on an array `(..., D)`.
      - `mode='full'`: retains all `D` channels unchanged.
      - `mode='mit'` : strictly zeros all AlphaGenome 1D/3D channels (`0%` AlphaGenome tracks).

    Returns `(masked_features, keep_mask)` where `keep_mask` has boolean shape `(D,)`.
    """
    cfg = get_mode_config(mode)
    arr = np.asarray(features, dtype=np.float32).copy()
    if arr.shape[-1] != len(feature_names):
        raise ValueError(
            f"Feature last dimension ({arr.shape[-1]}) does not match len(feature_names) ({len(feature_names)})"
        )
    if cfg.uses_alphagenome:
        keep_mask = np.ones(len(feature_names), dtype=bool)
        return arr, keep_mask

    keep_mask = np.array([not is_alphagenome_feature_name(nm) for nm in feature_names], dtype=bool)
    arr[..., ~keep_mask] = 0.0
    return arr, keep_mask


def mask_graph_for_mode(
    nodes: np.ndarray,
    node_mask: np.ndarray,
    edges: np.ndarray,
    schema: str = "ase",
    mode: str = "mit",
    node_act: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Mask AlphaGenome channels and peak coordinates on breakpoint-graph tensors in open-source mode:
      - `schema='ase'`     (`M=32, Dn>=24, De>=12`): masks out AlphaGenome peak-called enhancer
        slots `7..22` (`node_mask[..., 7:23] = False`), zeros AG chromatin/RNA/splice columns
        `6..21`, and zeros all 3D Hi-C/Micro-C/CTCF/ABC edge channels (`3,4,5,6,7,8,10,11`).
      - `schema='crispri'` (`M=32, Dn>=24, De>=12`): masks out AlphaGenome peak-called background
        slots `3..M-1` (`node_mask[..., 3:] = False`), zeros AG columns `6..19`, and zeros 3D
        edge channels `4..De-1`.
      - `schema='splice'`  (`M=18, Dn>=16, De>=10`): retains strictly geometric + Pangolin/Flashzoi
        columns `[0, 12, 13, 14, 15]`, zeros AG 2D RoPE & 3D edge channels `[2, 9]`, and removes
        slot-17 AG splice residual from `node_act` when supplied.

    Guarantees `ag_channel_count == 0` and `ag_coord_slot_count == 0` whenever `mode == 'mit'`.
    """
    cfg = get_mode_config(mode)
    n_out = np.asarray(nodes, dtype=np.float32).copy()
    m_out = np.asarray(node_mask, dtype=bool).copy()
    e_out = np.asarray(edges, dtype=np.float32).copy()
    act_out = np.asarray(node_act, dtype=np.float32).copy() if node_act is not None else None
    sc = str(schema).lower()
    if sc not in ("ase", "crispri", "splice"):
        raise ValueError(f"Unsupported graph schema {schema!r}; expected 'ase', 'crispri', or 'splice'")

    if cfg.uses_alphagenome:
        em = m_out[..., :, None] & m_out[..., None, :]
        res: Dict[str, Any] = {
            "nodes": n_out,
            "node_mask": m_out,
            "edges": e_out,
            "edge_mask": em,
            "mode": cfg.name,
            "schema": sc,
            "ag_channel_count": int(n_out.shape[-1]),
            "ag_coord_slot_count": int(m_out.sum()),
        }
        if act_out is not None:
            res["node_act"] = act_out
        return res

    M = n_out.shape[-2]
    Dn = n_out.shape[-1]
    De = e_out.shape[-1]

    if sc == "crispri":
        # Mask out AlphaGenome-peak-called background enhancer/CTCF slots (3..M-1)
        if M > 3:
            m_out[..., 3:] = False
        # Keep only node-type one-hot [0..4], surv_frac [5], Flashzoi [20..22], dist [23]
        keep_n = {c for c in (0, 1, 2, 3, 4, 5, 20, 21, 22, 23) if c < Dn}
        for c in range(Dn):
            if c not in keep_n:
                n_out[..., c] = 0.0
        # Keep strictly 1D distance / order edge channels [0..3]
        if De > 4:
            e_out[..., 4:] = 0.0
        if act_out is not None and M > 3:
            act_out[..., 3:, :] = 0.0

    elif sc == "ase":
        # Mask out AlphaGenome-peak-called enhancer/CTCF slots (7..22)
        hi_enh = min(23, M)
        if M > 7:
            m_out[..., 7:hi_enh] = False
        # Zero all AlphaGenome chromatin/RNA/splice node columns (6..21)
        for c in range(6, min(22, Dn)):
            n_out[..., c] = 0.0
        # Slot-specific cleanup for cols 22 and 23:
        # col 22 holds AG splice_lcl_usage on slots 0..6 (zero it!), SV len geometry on slots 23..31 (keep)
        if Dn > 22:
            n_out[..., : min(23, M), 22] = 0.0
        # col 23 holds Flashzoi RNA on TSS/Exons (0..6) & in_gene_body geometry on InsBub (23..28),
        # but AG abc_frac_gm on slots 7..22 (zero it!)
        if Dn > 23 and M > 7:
            n_out[..., 7:hi_enh, 23] = 0.0
        # Keep strictly 1D distance/order/breakpoint-crossing edges: [0, 1, 2, 9]
        keep_e = {c for c in (0, 1, 2, 9) if c < De}
        for c in range(De):
            if c not in keep_e:
                e_out[..., c] = 0.0
        if act_out is not None and M > 7:
            act_out[..., 7:hi_enh, :] = 0.0

    elif sc == "splice":
        # Keep strictly geometric [0] + Pangolin/Flashzoi [12, 13, 14, 15]
        keep_n = {c for c in (0, 12, 13, 14, 15) if c < Dn}
        for c in range(Dn):
            if c not in keep_n:
                n_out[..., c] = 0.0
        # Zero AG 2D RoPE [2] and AG 3D contact [9] edges
        for c in (2, 9):
            if c < De:
                e_out[..., c] = 0.0
        # Overwrite slot-17 AlphaGenome residual in node_act with pure geometric deletion survival
        if act_out is not None and M > 17:
            act_out[..., 17, 0] = (
                0.25
                * (1.0 - n_out[..., 0, 0])
                * n_out[..., 17, 0]
                * m_out[..., 17].astype(act_out.dtype)
            )

    # Zero masked-out nodes and edges for strict zero-leakage hygiene
    n_out = np.where(m_out[..., None], n_out, 0.0).astype(np.float32)
    em = m_out[..., :, None] & m_out[..., None, :]
    e_out = np.where(em[..., None], e_out, 0.0).astype(np.float32)
    if act_out is not None:
        act_out = np.where(m_out[..., None], act_out, 0.0).astype(np.float32)

    res = {
        "nodes": n_out,
        "node_mask": m_out,
        "edges": e_out,
        "edge_mask": em,
        "mode": cfg.name,
        "schema": sc,
        "ag_channel_count": 0,
        "ag_coord_slot_count": 0,
    }
    if act_out is not None:
        res["node_act"] = act_out
    return res


# ---------------------------------------------------------------------------
# Official Track Index Catalogs
# ---------------------------------------------------------------------------

# Official AlphaGenome Homo sapiens GM12878 (EFO:0002784) channel indices
AG_GM12878_RNA_CHANNELS = [132, 403, 573]           # polyA+ (+), polyA+ (-), polyA+ (unstranded)
AG_GM12878_CAGE_CHANNELS = [156, 429]               # hCAGE (+), hCAGE (-)
AG_GM12878_ATAC_CHANNEL = 63                        # EFO:0002784 ATAC-seq
AG_GM12878_DNASE_CHANNEL = 137                      # EFO:0002784 DNase-seq
AG_GM12878_HISTONE_CHANNELS = [446, 447, 448, 450]  # H3K27ac, H3K36me3, H3K4me1, H3K4me3
AG_GM12878_SPLICE_USAGE_CHANNELS = [153, 520]       # GM12878 polyA+ usage (+), (-)
AG_GM12878_CONTACT_CHANNEL = 4                      # 4dn:4DNFIXP4QG5B GM12878 Hi-C/Micro-C

# Official Calico Borzoi / Flashzoi GM12878 channel indices (from targets_human.txt)
BZ_GM12878_RNA_PLUS = [6112, 6114, 6116, 6118, 6204, 6514, 6668, 7306]
BZ_GM12878_RNA_MINUS = [6113, 6115, 6117, 6119, 6205, 6515, 6669, 7307]
BZ_GM12878_RNA_UNSTR = [6120, 6121, 6340, 6434, 7334, 7394]
BZ_GM12878_CAGE_PLUS = 870
BZ_GM12878_CAGE_MINUS = 871
BZ_GM12878_DNASE = [1288, 1345]
BZ_GM12878_H3K27AC = 2186
BZ_GM12878_H3K4ME3 = 2184

_ACGT_LUT = np.full(128, -1, dtype=np.int8)
for _idx, _b in enumerate("ACGT"):
    _ACGT_LUT[ord(_b)] = _idx
    _ACGT_LUT[ord(_b.lower())] = _idx


def encode_sequences_nlc(seqs: List[str], device: Any) -> Any:
    """Fast vectorized ASCII -> one-hot `(B, L, 4)` float32 tensor on target device."""
    B = len(seqs)
    L = len(seqs[0])
    arr = np.zeros((B, L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        b_arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        codes = _ACGT_LUT[np.clip(b_arr, 0, 127)]
        valid = codes >= 0
        arr[i, np.nonzero(valid)[0], codes[valid]] = 1.0
    return torch.from_numpy(arr).to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# GPU Model Runners (`full` and `mit` modes)
# ---------------------------------------------------------------------------


class AlphaGenomeG4Runner:
    """
    Frozen DeepMind AlphaGenome-450M runner (`gtca/alphagenome_pytorch`) with On-GPU
    Breakpoint-Graph pooling, full-context windows (`131kb / 262kb / 524kb / 1Mb`),
    optional FWD+RC averaging, and optional diagonal-normalized physical 3D contact `P_phys`.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda:0",
        mixed_precision: bool = True,
        window_len: int = 524288,
        rc_average: bool = True,
        use_physical_3d: bool = True,
    ):
        if window_len not in ALPHAGENOME_SUPPORTED_WINDOWS:
            raise ValueError(
                f"AlphaGenome window_len must be in {ALPHAGENOME_SUPPORTED_WINDOWS}, got {window_len}"
            )
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        self.window_len = int(window_len)
        self.rc_average = bool(rc_average)
        self.use_physical_3d = bool(use_physical_3d)
        policy = DtypePolicy.mixed_precision() if mixed_precision else DtypePolicy.full_float32()
        self.model = AlphaGenome.from_pretrained(weights_path, dtype_policy=policy, device=str(self.device))
        self.model.eval()
        self.heads = (
            "rna_seq",
            "cage",
            "atac",
            "dnase",
            "chip_histone",
            "contact_maps",
            "splice_sites",
            "splice_site_usage",
        )

    def _forward_slice(self, seqs: List[str], use_bf16: bool) -> Dict[str, Any]:
        x_nlc = encode_sequences_nlc(seqs, self.device)
        B = x_nlc.shape[0]
        org_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=bool(use_bf16)):
            raw = self.model.forward(
                x_nlc,
                org_idx,
                resolutions=(1, 128),
                heads=self.heads,
                channels_last=True,
            )
        sliced = {
            "rna": raw["rna_seq"][1][:, :, AG_GM12878_RNA_CHANNELS].float(),
            "cage": raw["cage"][1][:, :, AG_GM12878_CAGE_CHANNELS].float(),
            "atac": raw["atac"][1][:, :, AG_GM12878_ATAC_CHANNEL].float(),
            "dnase": raw["dnase"][1][:, :, AG_GM12878_DNASE_CHANNEL].float(),
            "hist": raw["chip_histone"][128][:, :, AG_GM12878_HISTONE_CHANNELS].float(),
            "sp_probs": raw["splice_sites"]["probs"][:, :, :4].float(),
            "sp_usage": raw["splice_site_usage"]["predictions"][:, :, AG_GM12878_SPLICE_USAGE_CHANNELS].float(),
            "cmap": raw["contact_maps"][:, :, :, AG_GM12878_CONTACT_CHANNEL].float(),
        }
        del raw
        return sliced

    def infer_and_pool_batch(
        self,
        mappings: List[HaplotypeGraphMapping],
        exons_list: List[List[Tuple[int, int]]],
        return_locus_tracks_flags: Optional[List[bool]] = None,
        use_bf16: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        if use_bf16 is None:
            use_bf16 = self.mixed_precision and (self.device.type == "cuda")
        with torch.no_grad():
            seqs = [m.sequence for m in mappings]
            B = len(seqs)
            fwd = self._forward_slice(seqs, bool(use_bf16))

            if self.rc_average:
                rc = self._forward_slice([reverse_complement(s) for s in seqs], bool(use_bf16))
                rna_sub = 0.5 * (fwd["rna"] + rc["rna"][:, :, [1, 0, 2]].flip(1))
                cage_sub = 0.5 * (fwd["cage"] + rc["cage"][:, :, [1, 0]].flip(1))
                atac_sub = 0.5 * (fwd["atac"] + rc["atac"].flip(1))
                dnase_sub = 0.5 * (fwd["dnase"] + rc["dnase"].flip(1))
                hist_sub = 0.5 * (fwd["hist"] + rc["hist"].flip(1))
                sp_probs_sub = 0.5 * (fwd["sp_probs"] + rc["sp_probs"][:, :, [2, 3, 0, 1]].flip(1))
                sp_usage_sub = 0.5 * (fwd["sp_usage"] + rc["sp_usage"][:, :, [1, 0]].flip(1))
                cmap_sub = 0.5 * (fwd["cmap"] + rc["cmap"].flip(1).flip(2))
                del rc
            else:
                rna_sub = fwd["rna"]
                cage_sub = fwd["cage"]
                atac_sub = fwd["atac"]
                dnase_sub = fwd["dnase"]
                hist_sub = fwd["hist"]
                sp_probs_sub = fwd["sp_probs"]
                sp_usage_sub = fwd["sp_usage"]
                cmap_sub = fwd["cmap"]
            del fwd

            results: List[Dict[str, Any]] = []
            for i in range(B):
                sub_i = {
                    "rna_1bp": rna_sub[i],
                    "cage_1bp": cage_sub[i],
                    "atac_1bp": atac_sub[i],
                    "dnase_1bp": dnase_sub[i],
                    "histone_128bp": hist_sub[i],
                    "splice_probs_1bp": sp_probs_sub[i],
                    "splice_usage_1bp": sp_usage_sub[i],
                    "contact_gm12878": cmap_sub[i],
                }
                want_trk = bool(return_locus_tracks_flags[i]) if return_locus_tracks_flags else False
                results.append(
                    pool_alphagenome_on_gpu(
                        sub_i,
                        mappings[i],
                        exons_list[i],
                        return_locus_tracks=want_trk,
                        use_physical_3d=self.use_physical_3d,
                    )
                )
            return results


def _install_pytorch_sdpa_flash_mha_shim() -> None:
    """
    Provide a 100% weight-compatible (`strict=True`) PyTorch 2.x native FlashAttention / SDPA
    fallback for `flash_attn.modules.mha.MHA` when the compiled `flash_attn` wheel is unavailable
    (e.g., newer CUDA / PyTorch 2.x environments where `torch.nn.functional.scaled_dot_product_attention`
    natively dispatches to cuDNN FlashAttention).
    """
    import importlib.machinery
    import sys
    import types

    try:
        import transformers.modeling_utils  # noqa: F401
    except Exception:
        pass

    try:
        import flash_attn.modules.mha  # noqa: F401
        return
    except (ImportError, ValueError):
        pass

    import torch.nn as nn
    import torch.nn.functional as F

    class _NativeSDPAMHA(nn.Module):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_heads_kv: Optional[int] = None,
            qkv_proj_bias: bool = True,
            out_proj_bias: bool = True,
            dropout: float = 0.0,
            softmax_scale: Optional[float] = None,
            causal: bool = False,
            rotary_emb_dim: int = 0,
            rotary_emb_base: float = 10000.0,
            rotary_emb_interleaved: bool = False,
            **kwargs: Any,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
            self.head_dim = embed_dim // num_heads
            self.softmax_scale = softmax_scale
            self.causal = causal
            self.dropout = dropout
            self.rotary_emb_dim = rotary_emb_dim
            self.rotary_emb_base = float(rotary_emb_base)
            self.rotary_emb_interleaved = rotary_emb_interleaved

            qkv_dim = embed_dim + 2 * self.num_heads_kv * self.head_dim
            self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

            if self.rotary_emb_dim > 0:
                inv_freq = 1.0 / (
                    self.rotary_emb_base
                    ** (torch.arange(0, self.rotary_emb_dim, 2, dtype=torch.float32) / float(self.rotary_emb_dim))
                )
                self.register_buffer("_inv_freq", inv_freq, persistent=False)

        def _apply_rope(self, q: Any, k: Any) -> Tuple[Any, Any]:
            S = q.shape[2]
            rd = self.rotary_emb_dim
            t = torch.arange(S, device=q.device, dtype=torch.float32)
            freqs = torch.outer(t, self._inv_freq.to(q.device))
            cos = torch.cos(freqs).to(q.dtype)[None, None, :, :]
            sin = torch.sin(freqs).to(q.dtype)[None, None, :, :]

            def _rot(x: Any) -> Any:
                x_rot, x_pass = x[..., :rd], x[..., rd:]
                if self.rotary_emb_interleaved:
                    x1 = x_rot[..., 0::2]
                    x2 = x_rot[..., 1::2]
                    xr = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
                else:
                    half = rd // 2
                    x1, x2 = x_rot[..., :half], x_rot[..., half:]
                    xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
                return torch.cat([xr, x_pass], dim=-1)

            return _rot(q), _rot(k)

        def forward(self, x: Any, **kwargs: Any) -> Any:
            B, S, _ = x.shape
            qkv = self.Wqkv(x)
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_heads_kv * self.head_dim
            q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)
            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_heads_kv, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_heads_kv, self.head_dim).transpose(1, 2)

            if self.rotary_emb_dim > 0:
                q, k = self._apply_rope(q, k)

            drop_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=drop_p,
                is_causal=self.causal,
                scale=self.softmax_scale,
                enable_gqa=(self.num_heads != self.num_heads_kv),
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
            return self.out_proj(attn_out)

    fa_pkg = types.ModuleType("flash_attn")
    fa_pkg.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    fa_pkg.__path__ = []  # type: ignore[attr-defined]
    fa_mods = types.ModuleType("flash_attn.modules")
    fa_mods.__spec__ = importlib.machinery.ModuleSpec("flash_attn.modules", loader=None, is_package=True)
    fa_mods.__path__ = []  # type: ignore[attr-defined]
    fa_mha = types.ModuleType("flash_attn.modules.mha")
    fa_mha.__spec__ = importlib.machinery.ModuleSpec("flash_attn.modules.mha", loader=None)
    fa_mha.MHA = _NativeSDPAMHA  # type: ignore[attr-defined]
    fa_mods.mha = fa_mha  # type: ignore[attr-defined]
    fa_pkg.modules = fa_mods  # type: ignore[attr-defined]
    sys.modules.setdefault("flash_attn", fa_pkg)
    sys.modules.setdefault("flash_attn.modules", fa_mods)
    sys.modules.setdefault("flash_attn.modules.mha", fa_mha)


class _BaseBorzoiFamilyRunner:
    """Shared frozen 524,288-bp runner for both `Borzoi-524kb` and `Flashzoi-524kb`."""

    def __init__(
        self,
        repo_id: str,
        device: str = "cuda:0",
        unsquash: bool = True,
        rc_average: bool = True,
        rna_scale: float = 0.30,
    ):
        _install_pytorch_sdpa_flash_mha_shim()
        from borzoi_pytorch import Borzoi
        from borzoi_pytorch.config_borzoi import BorzoiConfig
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.repo_id = repo_id
        self.device = torch.device(device)
        self.unsquash = bool(unsquash)
        self.rc_average = bool(rc_average)
        self.rna_scale = float(rna_scale)
        try:
            self.model = Borzoi.from_pretrained(repo_id).to(self.device)
        except Exception:
            cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            wt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg_dict = json.load(fh)
            self.model = Borzoi(BorzoiConfig(**cfg_dict)).to(self.device)
            sd = load_file(wt_path, device=str(self.device))
            self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.crop_offset_bp = (524288 - 196608) // 2  # 163,840 bp
        self.out_bp = 196608
        self.bin_bp = 32

    def _predict_raw_tracks(self, seqs: List[str], use_bf16: bool) -> Dict[str, Any]:
        x_f = encode_sequences_nlc(seqs, self.device).transpose(1, 2).contiguous()
        use_amp = bool(use_bf16) and (self.device.type == "cuda")
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            of = self.model(x_f, is_human=True).float()
        if self.rc_average:
            x_r = encode_sequences_nlc([reverse_complement(s) for s in seqs], self.device).transpose(1, 2).contiguous()
            with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                or_ = self.model(x_r, is_human=True).float()
            rp_sq = 0.5 * (of[:, BZ_GM12878_RNA_PLUS, :] + or_[:, BZ_GM12878_RNA_MINUS, :].flip(-1))
            rm_sq = 0.5 * (of[:, BZ_GM12878_RNA_MINUS, :] + or_[:, BZ_GM12878_RNA_PLUS, :].flip(-1))
            ru_sq = 0.5 * (of[:, BZ_GM12878_RNA_UNSTR, :] + or_[:, BZ_GM12878_RNA_UNSTR, :].flip(-1))
            cage_p = 0.5 * (of[:, BZ_GM12878_CAGE_PLUS, :] + or_[:, BZ_GM12878_CAGE_MINUS, :].flip(-1))
            cage_m = 0.5 * (of[:, BZ_GM12878_CAGE_MINUS, :] + or_[:, BZ_GM12878_CAGE_PLUS, :].flip(-1))
            dnase = 0.5 * (of[:, BZ_GM12878_DNASE, :] + or_[:, BZ_GM12878_DNASE, :].flip(-1)).mean(dim=1)
            h3k27ac = 0.5 * (of[:, BZ_GM12878_H3K27AC, :] + or_[:, BZ_GM12878_H3K27AC, :].flip(-1))
            h3k4me3 = 0.5 * (of[:, BZ_GM12878_H3K4ME3, :] + or_[:, BZ_GM12878_H3K4ME3, :].flip(-1))
            del or_
        else:
            rp_sq = of[:, BZ_GM12878_RNA_PLUS, :]
            rm_sq = of[:, BZ_GM12878_RNA_MINUS, :]
            ru_sq = of[:, BZ_GM12878_RNA_UNSTR, :]
            cage_p = of[:, BZ_GM12878_CAGE_PLUS, :]
            cage_m = of[:, BZ_GM12878_CAGE_MINUS, :]
            dnase = of[:, BZ_GM12878_DNASE, :].mean(dim=1)
            h3k27ac = of[:, BZ_GM12878_H3K27AC, :]
            h3k4me3 = of[:, BZ_GM12878_H3K4ME3, :]
        del of

        if self.unsquash:
            rna_p = undo_squashed_scale(rp_sq, scale=self.rna_scale).mean(dim=1)
            rna_m = undo_squashed_scale(rm_sq, scale=self.rna_scale).mean(dim=1)
            rna_u = undo_squashed_scale(ru_sq, scale=self.rna_scale).mean(dim=1)
        else:
            rna_p = rp_sq.mean(dim=1)
            rna_m = rm_sq.mean(dim=1)
            rna_u = ru_sq.mean(dim=1)

        return {
            "rna_plus": rna_p,
            "rna_minus": rna_m,
            "rna_unstr": rna_u,
            "cage_plus": cage_p,
            "cage_minus": cage_m,
            "dnase": dnase,
            "h3k27ac": h3k27ac,
            "h3k4me3": h3k4me3,
        }

    def infer_and_pool_batch(
        self,
        mappings: List[HaplotypeGraphMapping],
        exons_list: List[List[Tuple[int, int]]],
        use_bf16: bool = True,
    ) -> List[Dict[str, np.ndarray]]:
        with torch.no_grad():
            seqs = [m.sequence for m in mappings]
            tracks = self._predict_raw_tracks(seqs, use_bf16=use_bf16)

            results: List[Dict[str, np.ndarray]] = []
            for i, mapping in enumerate(mappings):
                sub_i = {k: v[i] for k, v in tracks.items()}
                results.append(
                    pool_borzoi_on_gpu(
                        sub_i,
                        mapping,
                        exons_list[i],
                        crop_offset_bp=self.crop_offset_bp,
                        out_bp=self.out_bp,
                    )
                )
            return results


class BorzoiG4Runner(_BaseBorzoiFamilyRunner):
    """Frozen Calico Borzoi-524kb runner (`johahi/borzoi-replicate-0`)."""

    def __init__(
        self,
        repo_id: str = "johahi/borzoi-replicate-0",
        device: str = "cuda:0",
        unsquash: bool = True,
        rc_average: bool = True,
    ):
        super().__init__(repo_id=repo_id, device=device, unsquash=unsquash, rc_average=rc_average)


class FlashzoiG4Runner(_BaseBorzoiFamilyRunner):
    """Frozen Flashzoi-524kb FlashAttention runner (`johahi/flashzoi-replicate-0`)."""

    def __init__(
        self,
        repo_id: str = "johahi/flashzoi-replicate-0",
        device: str = "cuda:0",
        unsquash: bool = True,
        rc_average: bool = True,
    ):
        super().__init__(repo_id=repo_id, device=device, unsquash=unsquash, rc_average=rc_average)


class Flashzoi4xEnsembleRunner:
    """
    4-replicate Flashzoi-524kb ensemble runner (MIT-licensed weights)
    (`johahi/flashzoi-replicate-{0,1,2,3}` with FWD+RC averaging & exact `undo_squashed_scale`).
    """

    def __init__(
        self,
        repo_ids: Sequence[str] = FLASHZOI_REPLICATE_REPOS,
        device: str = "cuda:0",
        unsquash: bool = True,
        rc_average: bool = True,
    ):
        self.repo_ids = tuple(repo_ids)
        self.device = device
        self.unsquash = bool(unsquash)
        self.rc_average = bool(rc_average)
        self.runners = [
            FlashzoiG4Runner(
                repo_id=rid,
                device=device,
                unsquash=self.unsquash,
                rc_average=self.rc_average,
            )
            for rid in self.repo_ids
        ]

    def infer_and_pool_batch(
        self,
        mappings: List[HaplotypeGraphMapping],
        exons_list: List[List[Tuple[int, int]]],
        use_bf16: bool = True,
    ) -> List[Dict[str, np.ndarray]]:
        with torch.no_grad():
            seqs = [m.sequence for m in mappings]
            n_reps = float(len(self.runners))
            acc: Optional[Dict[str, Any]] = None
            for runner in self.runners:
                tr = runner._predict_raw_tracks(seqs, use_bf16=use_bf16)
                if acc is None:
                    acc = {k: v / n_reps for k, v in tr.items()}
                else:
                    for k in acc:
                        acc[k] = acc[k] + tr[k] / n_reps

            assert acc is not None
            ref_r = self.runners[0]
            results: List[Dict[str, np.ndarray]] = []
            for i, mapping in enumerate(mappings):
                sub_i = {k: v[i] for k, v in acc.items()}
                results.append(
                    pool_borzoi_on_gpu(
                        sub_i,
                        mapping,
                        exons_list[i],
                        crop_offset_bp=ref_r.crop_offset_bp,
                        out_bp=ref_r.out_bp,
                    )
                )
            return results


class Pangolin12xRunner:
    """
    Open-source `Pangolin-12x` splicing ensemble runner (4 tissues x 3 replicates; `0%` AlphaGenome).
    Requires optional `torch` and `pangolin` packages at instantiation time.
    """

    def __init__(self, model_nums: Sequence[int] = (0, 2, 4, 6), device: str = "cuda:0"):
        if torch is None:
            raise ImportError("PyTorch is required for Pangolin12xRunner.")
        from pangolin.model import L, W, AR, Pangolin  # type: ignore[import-not-found]
        from pkg_resources import resource_filename

        self.device = torch.device(device)
        self.model_nums = tuple(model_nums)
        self.models: List[Any] = []
        for mn in self.model_nums:
            for rep in range(1, 4):
                m = Pangolin(L, W, AR).to(self.device)
                wt_p = resource_filename("pangolin", f"models/final.{rep}.{mn}.3")
                state = torch.load(wt_p, map_location=self.device)
                m.load_state_dict(state)
                m.eval()
                self.models.append(m)

    def predict_usage(self, seqs: List[str]) -> np.ndarray:
        """Predict mean splice-site usage probability across the 12 Pangolin checkpoints."""
        with torch.no_grad():
            x_ncl = encode_sequences_nlc(seqs, self.device).transpose(1, 2).contiguous()
            acc = None
            for m in self.models:
                out = m(x_ncl)[:, 1, :].float()
                acc = out if acc is None else (acc + out)
            assert acc is not None
            return (acc / float(len(self.models))).detach().cpu().numpy()


def measure_fp32_vs_bf16_parity(
    fp32_nodes: np.ndarray,
    bf16_nodes: np.ndarray,
) -> Dict[str, float]:
    """Quantify numerical parity between full FP32 and BF16 mixed-precision pooled Breakpoint-Graph nodes."""
    a = np.asarray(fp32_nodes, dtype=np.float64).ravel()
    b = np.asarray(bf16_nodes, dtype=np.float64).ravel()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-4)
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-8 and np.std(b) > 1e-8 else 1.0
    return {
        "pearson_r": round(corr, 6),
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "median_rel_diff": float(np.median(diff / denom)),
    }
