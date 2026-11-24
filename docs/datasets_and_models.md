# Public Datasets and Pretrained Checkpoints

All analyses in the accompanying manuscript rely on publicly available human reference cohorts and pretrained sequence-to-function model checkpoints. No large sequencing files or model weights are stored in this repository.

## Public Reference Cohorts

| Dataset | Role in Study | Access / Accession |
| :--- | :--- | :--- |
| 1KG_ONT_VIENNA | Long-read SV catalog on T2T-CHM13v2.0 (`hs1`) | Public 1000 Genomes ONT Consortium release |
| HPRC Releases 1 & 2 | Diploid phased assemblies and physical SV validation | Human Pangenome Reference Consortium (`HPRC`) |
| HGSVC3 | Phased long-read structural variants and trio validation | Human Genome Structural Variation Consortium |
| NYGC 1000 Genomes (30x) | Pedigree-phased GRCh38 SNV scaffolds for LD phase registration | NYGC 1kGP high-coverage release |
| MAGE (v1.0) | Personalized diploid lymphoblastoid RNA-seq, LeafCutter splicing, rare outliers | ENA `PRJNA851328` |
| Geuvadis | Phased exonic allele-specific expression counts | ArrayExpress `E-GEUV-1` |
| GTEx (v8 / v10) | 49-tissue eQTL summary statistics, SuSiE fine-mapping, Metasoft results | GTEx Portal (`phs000424.v10.p2`) |
| Chiang et al. (2017) | Primary human tissue deletion and duplication SV-eQTLs | Published supplementary dataset |
| ClinVar | Pathogenic and benign structural variants (>= 50 bp, GRCh38) | NCBI ClinVar GRCh38 release |
| ENCODE-rE2G | CRISPRi enhancer-gene perturbation screens (K562, GM12878) | ENCODE-rE2G public benchmark |
| 4D Nucleome | Reference Hi-C and Micro-C chromatin contact maps | `4DNFI1UEG1HD`, `4DNFI5IHU27G` |
| ENCODE4 lentiMPRA | Lentiviral massively parallel reporter assay indels | `ENCFF368WLS`, `ENCFF400HXX`, `ENCFF409FRY` |
| GENCODE v44 | Gene and exon annotations (GRCh38 and T2T-CHM13v2.0) | GENCODE public release |

## Pretrained Model Checkpoints and Licenses

The `panprism` codebase itself is released under the MIT License (`LICENSE`). Pretrained neural checkpoints remain governed by their upstream licenses:

| Model | Context Length | Upstream Source | Weight License | Used in Default Open-Source Mode (`--mode mit`)? |
| :--- | :---: | :--- | :--- | :---: |
| Flashzoi-524kb (4 replicates) | 524,288 bp | `johahi/flashzoi-replicate-{0..3}` | MIT | Yes |
| Pangolin-12x | 10,000 bp flank | `tkzeng/Pangolin` | GPL-3.0 | Yes |
| AlphaGenome-450M | 262 kb / 524 kb / 1 Mb | `google-deepmind/alphagenome` / `gtca/alphagenome_pytorch` | Non-Commercial Research Terms | No (`--mode full` only) |
| Borzoi-524kb (comparison) | 524,288 bp | `calico/borzoi` | Apache-2.0 / MIT | No (benchmark comparison) |
| OpenSpliceAI-5x (comparison) | 10,000 bp flank | `Kuanhao-Chao/OpenSpliceAI` | MIT | No (benchmark comparison) |

In default open-source mode (`--mode mit`), `panprism` masks all AlphaGenome channels and peak-called coordinates to zero so inference depends solely on Flashzoi-4x and Pangolin-12x.
