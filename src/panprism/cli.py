"""
Command-line interface for panprism.

  - `panprism verify-assembly` : Two-tier `V_15` + `DASG (V_graft)` hs1 <-> GRCh38 syntenic validator (VCF or GFA)
  - `panprism register-phase`  : DNA-only diploid SNV-SV LD scaffold phase registration
  - `panprism infer`           : Dual-mode (open-source default or full ensemble) breakpoint-graph S2F inference
  - `panprism couple`          : Parity-invariant diploid Biophysical Antisymmetric Ridge & 22-Chr LOCO / Cross-Ancestry evaluator
  - `panprism finemap`         : Cross-ancestry rare & singleton (`MAF < 1%`) SV expression outlier causal fine-mapper
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from panprism import __version__
from panprism.assembly_validator import classify_sv_allele, verify_and_liftover_sv
from panprism.breakpoint_graph import (
    HaplotypeGraphMapping,
    VariantAllele,
    build_haplotype_graph_sequence,
    compute_compression_telemetry,
    haplotype_sha1,
)
from panprism.coupler import (
    PANPRISM_CHANNEL_NONNEG_MASK,
    panprism_12channel_features,
    safe_corr,
    zero_shot_panprism_score,
)
from panprism.dasg import verify_and_graft_sv_two_tier
from panprism.finemap import finemap_outlier_windows
from panprism.gfa_parser import parse_gfa_bubbles
from panprism.model_selection import fit_grouped_antisymmetric_ridge
from panprism.phase_registration import register_sv_phase_scaffolds
from panprism.s2f_runners import build_v2_runner_spec


def _load_fasta_dict(fasta_path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    cur = None
    op = gzip.open if str(fasta_path).endswith(".gz") else open
    with op(fasta_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


def cmd_verify_assembly(args: argparse.Namespace) -> int:
    ref_dict = _load_fasta_dict(Path(args.fasta))
    hs1_dict = _load_fasta_dict(Path(args.hs1_fasta)) if getattr(args, "hs1_fasta", None) else None
    chain_obj = None
    if args.chain:
        from pyliftover import LiftOver

        chain_obj = LiftOver(str(args.chain))

    in_path = Path(args.input)
    rows = []
    if str(in_path).endswith((".gfa", ".gfa.gz")):
        op = gzip.open if str(in_path).endswith(".gz") else open
        with op(in_path, "rt", encoding="utf-8") as fh:
            bubbles = parse_gfa_bubbles(fh, ref_sample_prefix=args.ref_prefix)
        chrom = next(iter(ref_dict.keys()))
        for hap_id, vlist in bubbles.items():
            for v in vlist:
                chk = verify_and_liftover_sv(
                    chrom, v.pos0, v.ref, v.alt, ref_dict[chrom], liftover_chain=None, flank_bp=args.flank_bp
                )
                rows.append(
                    {
                        "sample_hap": hap_id,
                        "var_id": v.var_id,
                        "sv_class": v.sv_class,
                        "sv_len": v.sv_len,
                        "pos0_grch38": chk.pos0_grch38,
                        "tier": "tier1_v15" if chk.passed else "failed",
                        "passed": chk.passed,
                        "failure_reason": chk.failure_reason,
                    }
                )
    else:
        op = gzip.open if str(in_path).endswith(".gz") else open
        with op(in_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                chrom, pos1_s, vid, ref, alt = p[0], p[1], p[2], p[3], p[4].split(",")[0]
                if chrom not in ref_dict:
                    continue
                info = p[7] if len(p) > 7 else ""
                sv_cls, sv_len = classify_sv_allele(vid, ref, alt, info)
                if hs1_dict is not None and chrom in hs1_dict and chain_obj is not None:
                    res2 = verify_and_graft_sv_two_tier(
                        chrom=chrom,
                        pos0_source=int(pos1_s) - 1,
                        ref_allele=ref,
                        alt_allele=alt,
                        grch38_chrom_seq=ref_dict[chrom],
                        hs1_chrom_seq=hs1_dict[chrom],
                        liftover_chain=chain_obj,
                        flank_bp=args.flank_bp,
                        max_search_bp=args.dasg_search_bp,
                    )
                    rows.append(
                        {
                            "chrom": chrom,
                            "var_id": vid,
                            "sv_class": sv_cls,
                            "sv_len": sv_len,
                            "pos0_source": res2.pos0_source,
                            "pos0_grch38": res2.pos0_grch38,
                            "end0_grch38": res2.end0_grch38,
                            "coord_offset_bp": res2.tier1_result.coord_offset_bp,
                            "naive_grch38_match": res2.tier1_result.naive_grch38_match,
                            "tier": res2.tier,
                            "anchor_mismatch_total": (
                                res2.dasg_result.anchor_mismatch_total if res2.dasg_result is not None else 0
                            ),
                            "passed": res2.passed,
                            "failure_reason": res2.failure_reason,
                        }
                    )
                else:
                    chk = verify_and_liftover_sv(
                        chrom,
                        int(pos1_s) - 1,
                        ref,
                        alt,
                        ref_dict[chrom],
                        liftover_chain=chain_obj,
                        flank_bp=args.flank_bp,
                    )
                    rows.append(
                        {
                            "chrom": chrom,
                            "var_id": vid,
                            "sv_class": sv_cls,
                            "sv_len": sv_len,
                            "pos0_source": chk.pos0_source,
                            "pos0_grch38": chk.pos0_grch38,
                            "coord_offset_bp": chk.coord_offset_bp,
                            "naive_grch38_match": chk.naive_grch38_match,
                            "tier": "tier1_v15" if chk.passed else "failed",
                            "passed": chk.passed,
                            "failure_reason": chk.failure_reason,
                        }
                    )

    df = pd.DataFrame(rows)
    summary = {
        "version": __version__,
        "dasg_enabled": bool(hs1_dict is not None and chain_obj is not None),
        "n_records_checked": int(len(df)),
        "n_passed": int(df["passed"].sum()) if len(df) else 0,
        "n_tier1_passed": int((df["tier"] == "tier1_v15").sum()) if len(df) else 0,
        "n_tier2_dasg_rescued": int((df["tier"] == "tier2_vgraft").sum()) if len(df) else 0,
        "pass_rate": round(float(df["passed"].mean()), 4) if len(df) else 0.0,
    }
    if "naive_grch38_match" in df.columns and len(df):
        summary["naive_unlifted_match_rate"] = round(float(df["naive_grch38_match"].mean()), 4)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"summary": summary, "records": rows[:500]}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_register_phase(args: argparse.Namespace) -> int:
    data = np.load(args.input_npz, allow_pickle=True)
    reg_h1, reg_h2, flips, summary = register_sv_phase_scaffolds(
        sv_chroms=[str(x) for x in data["sv_chroms"]],
        sv_gene_ids=[str(x) for x in data["sv_gene_ids"]],
        sv_pos=data["sv_pos"],
        sv_h1=data["sv_h1"],
        sv_h2=data["sv_h2"],
        snv_chroms=[str(x) for x in data["snv_chroms"]],
        snv_gene_ids=[str(x) for x in data["snv_gene_ids"]],
        snv_pos=data["snv_pos"],
        snv_h1=data["snv_h1"],
        snv_h2=data["snv_h2"],
        max_dist_bp=args.max_dist_bp,
    )
    out_p = Path(args.output_npz)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, reg_h1=reg_h1, reg_h2=reg_h2, chrom_donor_flips=flips)
    out_json = out_p.with_suffix(".summary.json")
    out_json.write_text(json.dumps(summary.__dict__, indent=2))
    print(json.dumps(summary.__dict__, indent=2))
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    eff_mode = args.mode
    if eff_mode is None:
        eff_mode = "full" if args.backbone == "alphagenome" else "mit"

    runner_spec = build_v2_runner_spec(
        mode=eff_mode,
        backbone=args.backbone,
        window_len=args.window_len,
        device=args.device,
    )
    window_len = int(runner_spec["window_len"])
    backbones = list(runner_spec["backbones"])

    spec = json.loads(Path(args.manifest_json).read_text())
    unique_map: Dict[str, HaplotypeGraphMapping] = {}
    unique_exons: Dict[str, List] = {}
    records_meta = []

    for rec in spec["loci"]:
        variants = [VariantAllele(**v) for v in rec.get("variants", [])]
        mp = build_haplotype_graph_sequence(
            ref_seq=rec["ref_seq"],
            ref_seq_start0=int(rec["ref_seq_start0"]),
            tss_pos0=int(rec["tss_pos0"]),
            strand=rec.get("strand", "+"),
            variants=variants,
            window_len=window_len,
        )
        key = haplotype_sha1(mp, gene_id=rec["gene_id"])
        unique_map[key] = mp
        unique_exons[key] = [tuple(x) for x in rec.get("exons0", [])]
        records_meta.append({"locus_id": rec.get("locus_id", rec["gene_id"]), "sha1": key})

    if args.dry_run:
        tel = compute_compression_telemetry(
            n_alphagenome_seqs=len(unique_map) if "alphagenome" in backbones else 0,
            n_borzoi_seqs=len(unique_map) if "borzoi" in backbones else 0,
            n_flashzoi_seqs=len(unique_map) if any(b in ("flashzoi", "flashzoi-4x") for b in backbones) else 0,
        )
        out_obj = {
            "version": __version__,
            "mode": runner_spec["mode"],
            "backbone": args.backbone,
            "backbones": backbones,
            "uses_alphagenome": runner_spec["uses_alphagenome"],
            "alphagenome_track_fraction": runner_spec["alphagenome_track_fraction"],
            "alphagenome_coord_fraction": runner_spec["alphagenome_coord_fraction"],
            "window_len": window_len,
            "n_input_loci": len(records_meta),
            "n_unique_haplotypes": len(unique_map),
            "telemetry": tel,
        }
        Path(args.output).write_text(json.dumps(out_obj, indent=2))
        print(json.dumps(out_obj, indent=2))
        return 0

    from panprism.s2f_runners import (
        AlphaGenomeG4Runner,
        BorzoiG4Runner,
        Flashzoi4xEnsembleRunner,
        FlashzoiG4Runner,
    )

    keys = list(unique_map.keys())
    primary_bb = backbones[0]
    if primary_bb == "alphagenome":
        runner = AlphaGenomeG4Runner(
            weights_path=args.weights,
            device=args.device,
            window_len=window_len,
            use_physical_3d=not args.no_physical_3d,
        )
    elif primary_bb == "borzoi":
        runner = BorzoiG4Runner(device=args.device)
    elif primary_bb == "flashzoi-4x":
        runner = Flashzoi4xEnsembleRunner(device=args.device)
    else:
        runner = FlashzoiG4Runner(device=args.device)

    preds: Dict[str, Dict] = {}
    bs = args.batch_size
    for i in range(0, len(keys), bs):
        sub_k = keys[i : i + bs]
        batch_out = runner.infer_and_pool_batch([unique_map[k] for k in sub_k], [unique_exons[k] for k in sub_k])
        for k, bo in zip(sub_k, batch_out):
            preds[k] = bo

    np.savez_compressed(
        args.output,
        keys=np.array(keys),
        mode=np.array([runner_spec["mode"]]),
        backbones=np.array(backbones),
        panprism_nodes=np.stack([preds[k]["panprism_nodes"] for k in keys]),
        korsakova_stitch=np.stack([preds[k]["korsakova_stitch"] for k in keys]),
        center_crop_naive=np.stack([preds[k]["center_crop_naive"] for k in keys]),
    )
    return 0


def cmd_couple(args: argparse.Namespace) -> int:
    d = np.load(args.input_npz, allow_pickle=True)
    delta_pp = d["delta_panprism"] if "delta_panprism" in d else d["delta_pb"]
    y = d["y_ase"]
    chroms = np.array([str(c) for c in d["chroms"]])

    X12 = panprism_12channel_features(delta_pp, mode="full")
    zs_pred = zero_shot_panprism_score(delta_pp, mode="full")
    loco_pred = np.zeros(len(y), dtype=np.float32)

    for ch in sorted(set(chroms)):
        te = chroms == ch
        tr = ~te
        if np.sum(tr) < 8 or np.sum(te) < 1:
            loco_pred[te] = zs_pred[te]
            continue
        fitted = fit_grouped_antisymmetric_ridge(X12[tr], y[tr], chroms[tr], PANPRISM_CHANNEL_NONNEG_MASK)
        loco_pred[te] = fitted.predict(X12[te])

    nz = np.abs(y) > 1e-8

    def _accuracy(pred):
        return (
            float(np.mean(np.where(np.abs(pred[nz]) <= 1e-12, 0.5, np.sign(pred[nz]) == np.sign(y[nz]))))
            if np.any(nz)
            else 0.5
        )

    zs_acc = _accuracy(zs_pred)
    loco_acc = _accuracy(loco_pred)
    loco_r, loco_rho = safe_corr(y[nz], loco_pred[nz])
    null_accuracies = []
    for seed in range(args.null_seeds):
        rng = np.random.default_rng(1000 + seed)
        null_x = X12[rng.permutation(len(y))] * rng.choice([-1.0, 1.0], size=(len(y), 1))
        null_pred = np.zeros(len(y))
        for ch in sorted(set(chroms)):
            te = chroms == ch
            tr = ~te
            if len(set(chroms[tr])) < 2:
                continue
            fitted = fit_grouped_antisymmetric_ridge(null_x[tr], y[tr], chroms[tr], PANPRISM_CHANNEL_NONNEG_MASK)
            null_pred[te] = fitted.predict(null_x[te])
        null_accuracies.append(_accuracy(null_pred))
    null_res = {
        "n_seeds": args.null_seeds,
        "direction_acc_mean": float(np.mean(null_accuracies)) if null_accuracies else None,
        "per_seed_direction_acc": null_accuracies,
        "selection": "chromosome-grouped inner CV, training-fold RMS",
    }

    summary = {
        "n_pairs": int(len(y)),
        "inner_cv": "chromosome-grouped, fold-specific RMS, alpha by MSE",
        "zero_shot_dir_acc": round(zs_acc, 4),
        "loco_biophysical_dir_acc": round(loco_acc, 4),
        "loco_pearson_r": round(loco_r, 4),
        "loco_spearman_rho": round(loco_rho, 4),
        "null_control": null_res,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_finemap(args: argparse.Namespace) -> int:
    df = (
        pd.read_parquet(args.input_parquet)
        if str(args.input_parquet).endswith(".parquet")
        else pd.read_csv(args.input_parquet, sep="\t")
    )
    evid, pips = finemap_outlier_windows(
        window_ids=df["window_id"].to_numpy(),
        signed_z_outlier=df["z_outlier"].to_numpy(dtype=np.float32),
        signed_s2f_delta=df["panprism_delta"].to_numpy(dtype=np.float32),
    )
    df = df.copy()
    df["panprism_evidence"] = evid
    df["panprism_relative_priority"] = pips
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if str(out_p).endswith(".parquet"):
        df.to_parquet(out_p, index=False)
    else:
        df.to_csv(out_p, sep="\t", index=False)
    top1 = df.sort_values("panprism_relative_priority", ascending=False).groupby("window_id").head(1)
    summary = {
        "n_windows": int(df["window_id"].nunique()),
        "n_candidates": int(len(df)),
        "mean_top1_relative_priority": round(float(top1["panprism_relative_priority"].mean()), 4),
        "interpretation": "Within-window heuristic priority; not a causal posterior probability",
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="panprism",
        description=f"PanPrism {__version__}: PanPrism predicts regulatory and splicing effects of structural variants across human pangenomes",
    )
    p.add_argument("--version", action="version", version=f"panprism {__version__}")
    sub = p.add_subparsers(dest="subcommand", required=True)

    p_ver = sub.add_parser(
        "verify-assembly",
        help="Validate VCF/GFA SV alleles via Tier-1 V_15 & optional Tier-2 DASG (V_graft) syntenic grafting",
    )
    p_ver.add_argument("--input", required=True, help="Input VCF (.vcf/.vcf.gz) or GFA (.gfa/.gfa.gz)")
    p_ver.add_argument("--fasta", required=True, help="GRCh38 reference FASTA")
    p_ver.add_argument("--hs1-fasta", default=None, help="Optional hs1/CHM13v2.0 FASTA enabling Tier-2 DASG (V_graft)")
    p_ver.add_argument("--chain", default=None, help="Optional UCSC LiftOver chain (e.g., hs1ToHg38.over.chain.gz)")
    p_ver.add_argument("--flank-bp", type=int, default=15)
    p_ver.add_argument("--dasg-search-bp", type=int, default=250, help="Tier-2 DASG outward anchor search radius (bp)")
    p_ver.add_argument("--ref-prefix", default="GRCh38")
    p_ver.add_argument("--output", required=True, help="Output JSON report")
    p_ver.set_defaults(func=cmd_verify_assembly)

    p_reg = sub.add_parser("register-phase", help="100%% RNA-free diploid SNV x SV LD scaffold polarity registration")
    p_reg.add_argument("--input-npz", required=True)
    p_reg.add_argument("--max-dist-bp", type=int, default=25000)
    p_reg.add_argument("--output-npz", required=True)
    p_reg.set_defaults(func=cmd_register_phase)

    p_inf = sub.add_parser(
        "infer",
        help="Run dual-mode (mit default: 0%% AlphaGenome; full: +AlphaGenome-524kb/1Mb) Breakpoint-Graph S2F inference",
    )
    p_inf.add_argument("--manifest-json", required=True)
    p_inf.add_argument(
        "--mode",
        choices=["mit", "full"],
        default=None,
        help="Inference license/backbone mode: 'mit' (default, 0%% AlphaGenome) or 'full' (AlphaGenome + Flashzoi-4x + Pangolin-12x)",
    )
    p_inf.add_argument(
        "--backbone",
        choices=["auto", "alphagenome", "borzoi", "flashzoi", "flashzoi-4x", "pangolin-12x"],
        default="auto",
    )
    p_inf.add_argument(
        "--window-len",
        type=int,
        default=None,
        help="Sequence window length in bp (default 524288; AlphaGenome full mode supports 131072/262144/524288/1048576)",
    )
    p_inf.add_argument("--weights", default="model_all_folds.safetensors")
    p_inf.add_argument("--device", default="cuda:0")
    p_inf.add_argument("--batch-size", type=int, default=12)
    p_inf.add_argument("--no-physical-3d", action="store_true", help="Disable diagonal-normalized P_phys 3D contact transform")
    p_inf.add_argument("--dry-run", action="store_true")
    p_inf.add_argument("--output", required=True)
    p_inf.set_defaults(func=cmd_infer)

    p_cpl = sub.add_parser("couple", help="Fit parity-invariant diploid Biophysical Antisymmetric Coupler (22-Chr LOCO)")
    p_cpl.add_argument("--input-npz", required=True)
    p_cpl.add_argument("--null-seeds", type=int, default=20)
    p_cpl.add_argument("--output", required=True)
    p_cpl.set_defaults(func=cmd_couple)

    p_fm = sub.add_parser(
        "finemap", help="Rank rare variant candidates in outlier windows; priorities are not causal probabilities"
    )
    p_fm.add_argument("--input-parquet", required=True)
    p_fm.add_argument("--output", required=True)
    p_fm.set_defaults(func=cmd_finemap)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
