# Analysis Scripts and Workflow Notes

This directory documents how the `panprism` command-line tools are combined to reproduce the analyses described in the manuscript.

## End-to-End Command Sequence

1. **Cross-assembly SV verification and syntenic grafting (`verify-assembly`)**
   ```bash
   panprism verify-assembly \
     --input long_read_svs.vcf.gz \
     --fasta GRCh38.fa \
     --hs1-fasta chm13v2.0.fa \
     --chain hs1ToHg38.over.chain.gz \
     --output verified_svs.json
   ```

2. **DNA-only haplotype phase registration (`register-phase`)**
   ```bash
   panprism register-phase \
     --input-npz cohort_sv_snv_genotypes.npz \
     --output-npz phase_registered_svs.npz
   ```

3. **Breakpoint-graph inference (`infer`)**
   ```bash
   # Default open-source mode (Flashzoi-4x + Pangolin-12x; no AlphaGenome weights)
   panprism infer --manifest-json loci.json --mode mit --output pooled_open.npz

   # Optional full multi-backbone mode (adds AlphaGenome 262 kb / 524 kb / 1 Mb + 3D contacts)
   panprism infer --manifest-json loci.json --mode full --window-len 524288 --output pooled_full.npz
   ```

4. **Chromosome-grouped allelic coupling (`couple`)**
   ```bash
   panprism couple \
     --input-npz het_sv_ase_features.npz \
     --null-seeds 20 \
     --output loco_ase_results.json
   ```

5. **Rare and singleton SV outlier prioritization (`finemap`)**
   ```bash
   panprism finemap \
     --input-parquet outlier_windows.parquet \
     --output prioritized_rare_svs.parquet
   ```

## Per-Stratum Benchmark Outputs

All summary tables and per-panel values underlying Figures 1–6 and Supplementary Figures 1–12 are provided in the manuscript Source Data and Supplementary Tables workbooks. See `docs/datasets_and_models.md` for accession codes of all public input cohorts.
