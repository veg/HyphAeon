<div align="center">

<img src="assets/hyphaeon_logo.png" alt="HyphAeon Logo" width="280"/>

# HyphAeon
### Attention on Evolution Across Deep Time Transforms Comparative Genomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.09.06.749597-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Interactive%20Presentation-00e5ff.svg)](https://veg.github.io/HyphAeon/)

</div>

---

**HyphAeon** is a deep-time phylogenetic foundation model designed to bridge computational phylogenetics, structural biology, and foundation AI. Built upon a 2D axial transformer backbone (**`PhyloAxialTransformer`**) with patristic distance-decay attention and classical multidimensional scaling (MDS) tree embeddings, HyphAeon ingests multi-species codon alignments and explicit evolutionary trees spanning 200 million years of deep time.

## 📦 Monorepo Structure

This repository is organized as a monorepo with three installable packages:

| Package | Description | Install |
| :--- | :--- | :--- |
| **`aeon-core`** | Shared model, dataset, inference, weights, stats, and IO infrastructure | `pip install hyphaeon-core` |
| **`hyphaeon`** | Site-level selection inference, epistasis, phenotype association, disease prediction | `pip install hyphaeon` |
| **`chronaeon`** | Molecular clock dating, phylodynamics, phylogeography, genomic surveillance | `pip install chronaeon` |

`aeon-core` is the shared foundation. Both `hyphaeon` and `chronaeon` depend on it
and pull it in automatically when installed. Example datasets are shared from the
root `examples/` directory; chronaeon-specific examples live in `chronaeon/examples/`.

---

## 🔭 The Radar & Microscope Flywheel

**ChronAeon** is the **Radar**: rapidly screens genomes, detects emerging clades, and infers origin dates. **HyphAeon** is the **Microscope**: dissects *why* flagged clades emerged — identifying positive selection bursts and epistatic rewiring. Together they form a collaborative flywheel for genomic surveillance and deep evolutionary analysis.

---

> [!TIP]
> **Migrating from HyPhy?** See our comprehensive [**HyPhy to HyphAeon Migration Guide**](hyphaeon/MIGRATION_GUIDE.md) for direct method-by-method translations (`hyphy meme` → `hyphaeon meme`, `contrast-fel` → `hyphaeon phenotype`, `prime` → `hyphaeon dms`) and biological recipes categorized by empirical data regime.

## 🚀 Capabilities at a Glance

### HyphAeon — The Microscope

Site-level selection inference, 3D epistasis, digital DMS, phenotype association, spectral bisection, and temporal dynamics. See [`hyphaeon/README.md`](hyphaeon/README.md) for detailed examples, CLI subcommands, and install options.

| Command | Description |
| :--- | :--- |
| `hyphaeon meme` | Neural episodic positive selection (100×–1,100× faster than MLE) |
| `hyphaeon epistasis` | 3D co-evolution & epistatic sector mining |
| `hyphaeon dms` | In silico deep mutational scanning & CPD prediction |
| `hyphaeon phenotype` | PhyloWAS phenotype-genotype association mapping |
| `hyphaeon splits` | Tree-free spectral graph bisection & clade discovery |
| `hyphaeon temporal` | Continuous surveillance dynamics & sweep velocity |
| `hyphaeon disease` | Disease pathogenicity prediction |
| `hyphaeon filter` | Automated alignment error screening & masking |

### ChronAeon — The Radar

Molecular clock dating, phylodynamics, phylogeography, streaming genomic triage, multi-clock deconvolution, and alignment-free binning. See [`chronaeon/README.md`](chronaeon/README.md) for detailed examples, CLI subcommands, and install options.

| Command | Description |
| :--- | :--- |
| `chronaeon date` | Heterochronous molecular clock calibration & t_MRCA dating |
| `chronaeon autoclock` | Hierarchical multi-clock community deconvolution |
| `chronaeon triage` | Streaming genomic QC triage / outbreak radar |
| `chronaeon phylogeo` | Discrete phylogeography & spatial transmission networks |
| `chronaeon dynamics` | Phylodynamic R₀/Rₜ estimation |
| `chronaeon sketch` | Alignment-free MinHash sketching & binning |
| `chronaeon align` | Reference-guided codon-aware alignment |

---

## 📦 Installation

Both packages require Python ≥ 3.8 and PyTorch ≥ 2.0. At runtime they auto-select
the best available device (CUDA → Apple MPS → CPU), so no manual configuration
is needed regardless of which install path you choose.

```bash
pip install hyphaeon    # The Microscope — selection, epistasis, DMS, phenotype
pip install chronaeon   # The Radar — dating, phylodynamics, phylogeography, triage
```

For detailed install options (CPU-only wheels, Bioconda, NVIDIA Jetson, model
weights, specific CUDA builds), see the respective package READMEs:
- [`hyphaeon/README.md`](hyphaeon/README.md#installation)
- [`chronaeon/README.md`](chronaeon/README.md#installation)

### Local Development

This is a monorepo with no root-level installable package. To set up a
development environment from a fresh clone:

```bash
pip install -e aeon-core -e hyphaeon -e chronaeon
```

Run the test suite per-subpackage (each has its own `conftest.py` and `pyproject.toml`):

```bash
pytest chronaeon/tests/    # ChronAeon tests
pytest hyphaeon/tests/     # HyphAeon tests
```

> [!NOTE]
> **Model weights** are downloaded automatically from [Hugging Face](https://huggingface.co/datamonkey/hyphaeon)
> on first use (cached in `~/.cache/hyphaeon/`). No authentication or token is
> required. Use `--model-variant viral` to select the viral-tuned variant, or
> `--weights /path/to/checkpoint` to use a local file.

---

## 📂 Included Benchmark Datasets

The repository bundles canonical historical and pandemic outbreak benchmarks in `examples/` and `chronaeon/examples/`:

| Dataset | Alignment | Tree / Coordinates | Taxa | Sites | Scientific Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HIV-1 RT** | [`examples/HIV1_RT.fasta`](examples/HIV1_RT.fasta) | [`examples/HIV1_RT.nwk`](examples/HIV1_RT.nwk) | 476 | 335 | Retroviral Reverse Transcriptase polymerase domain (drug resistance & epistasis). |
| **Rhodopsin** | [`examples/RHO.fasta`](examples/RHO.fasta) | Auto (TN93) | 710 | 349 | Mammalian Rhodopsin visual pigments (deep-sea diving sensory adaptation). No tree file provided; uses TN93 distance estimation. |
| **Smc6** | [`examples/Smc6.fasta`](examples/Smc6.fasta) | [`examples/Smc6.nwk`](examples/Smc6.nwk) | 20 | 1,097 | Primate Smc6 structural maintenance of chromosomes (antiviral host restriction). |
| **Bat OAS1** | [`examples/bat_oas1.fasta`](examples/bat_oas1.fasta) | [`examples/bat_oas1.nwk`](examples/bat_oas1.nwk) | 18 | 351 | Chiropteran OAS1 2'-5'-oligoadenylate synthetase (innate immunity escape). |
| **Camelid VHH** | [`examples/camelid.fasta`](examples/camelid.fasta) | [`examples/camelid.nwk`](examples/camelid.nwk) | 212 | 96 | Camelid single-domain antibody heavy-chain variable domain (antigenic diversity). Used for integration testing; no dedicated example section. |
| **HIV-1 gp160 (Korber 2000)** | [`chronaeon/examples/korber_env_gp160.fasta`](chronaeon/examples/korber_env_gp160.fasta) | Tree-Free / Consensus | 143 | 981 | Bette Korber et al. (Science 2000) landmark molecular clock dataset (1959–1997 HIV-1 group M). |
| **Avian Flu H5N1 (Lemey 2009)** | [`chronaeon/examples/H5N1_HA_geo.fasta`](chronaeon/examples/H5N1_HA_geo.fasta) | [`chronaeon/examples/H5N1_HA.nwk`](chronaeon/examples/H5N1_HA.nwk) | 98 | 566 | Lemey et al. (PLoS Comput Biol 2009) benchmark discrete phylogeography across 7 Chinese provinces. |
| **Pandemic H1N1 (Fraser 2009)** | [`chronaeon/examples/H1N1_2009_pandemic.fasta`](chronaeon/examples/H1N1_2009_pandemic.fasta) | [`chronaeon/examples/H1N1_2009_pandemic.nwk`](chronaeon/examples/H1N1_2009_pandemic.nwk) | 100 | 1701 | Fraser et al. (Science 2009) landmark phylodynamics and early growth rate benchmark ($R_0$ estimation). |

---

## 🔬 Reproducible Benchmark Examples

Reproducible worked examples with full CLI commands, output summaries, and
biological interpretations are available in each package's README:

- **HyphAeon examples** (epistasis, DMS, phenotype, selection, spectral splits):
  see [`hyphaeon/README.md`](hyphaeon/README.md#reproducible-benchmark-examples)
- **ChronAeon examples** (molecular clock dating, phylogeography, phylodynamics):
  see [`chronaeon/README.md`](chronaeon/README.md#reproducible-benchmark-examples)

---

## 📜 Citation

If you use **HyphAeon** in your research, please cite:

> Sergei L. Kosakovsky Pond, Steven Weaver, Danielle Callan, Jordan D. Zehr, Alexander G. Lucaci, Hannah Verdonk, Avery Selberg, Gallean Brown, Maria Chikina, Nathan L. Clark, Kateryna D. Makova, Darren P. Martin, and Anton Nekrutenko.  
> **HyphAeon: Attention on Evolution Across Deep Time Transforms Comparative Genomics**.  
> *bioRxiv* 2026.09.06.749597; doi: [https://doi.org/10.64898/2026.09.06.749597](https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1)

```bibtex
@article{kosakovskypond2026hyphaeon,
  title={HyphAeon: Attention on Evolution Across Deep Time Transforms Comparative Genomics},
  author={Kosakovsky Pond, Sergei L. and Weaver, Steven and Callan, Danielle and Zehr, Jordan D. and Lucaci, Alexander G. and Verdonk, Hannah and Selberg, Avery and Brown, Gallean and Chikina, Maria and Clark, Nathan L. and Makova, Kateryna D. and Martin, Darren P. and Nekrutenko, Anton},
  journal={bioRxiv},
  pages={2026.09.06.749597},
  year={2026},
  doi={10.64898/2026.09.06.749597},
  url={https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1}
}
```
