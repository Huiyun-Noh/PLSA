# A Unified Pairwise Distribution Matching Framework for Graph Domain Adaptation under Structure Shift

Reproduction code for the paper's CSBM (synthetic) and real-data (ACM/DBLP,
Twitch, CORA, Airport) experiments.

## Repository layout

```text
run.py          # Main CLI: csbm / csbm-calib / real / real-sweep
methods/        # Downstream GDA backbones (a2gnn, specreg, dgsda, sagda)
models/         # Reweighting / predictor modules (plsa, pairalign, strurw, ...)
utils/          # Data loading, calibration, metrics, graph utilities
data/           # Datasets (not included; see "Datasets" below)
```

## Installation

Developed and tested with Python 3.9.

```bash
conda create -n gda python=3.9 -y
conda activate gda
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1
pip install -r requirements.txt
```

(`torch`/`torch_geometric` are installed separately first since the correct
build depends on your CUDA version; swap `cu121` for your platform.)

## Datasets

```text
data/acm/raw/        acm_docs.txt   acm_edgelist.txt   acm_labels.txt
data/dblp/raw/       dblp_docs.txt  dblp_edgelist.txt  dblp_labels.txt
data/cora_data/      good_cora_src_data.npz   good_cora_src_adj.npz
                     good_cora_tar_data.npz   good_cora_tar_adj.npz
data/twitch/         twitch_DE.npz    twitch_ENGB.npz    twitch_ES.npz   ...
data/airport_data/   airport_usa.npz  airport_brazil.npz  airport_europe.npz
```

- **ACM / DBLP**: citation-network splits from Wu et al., *Unsupervised
  domain adaptive graph convolutional networks*, WWW 2020.
- **Twitch**: [Twitch Social Networks](https://snap.stanford.edu/data/twitch-social-networks.html).
- **CORA**: Bojchevski & Günnemann (2018), using the Degree-split from
  Gui et al., *GOOD: A graph OOD benchmark*, NeurIPS 2022.
- **Airport**: Zhu et al. (2021b) flight-connection networks,
  [github.com/GentleZhu/EGI](https://github.com/GentleZhu/EGI/tree/main/data).

Place the files as shown above; loader paths are anchored to `run.py`'s
location, so the working directory does not matter.

## Usage

**Quick sanity check** (single HP combo, seconds):

```bash
python run.py --device cuda real \
  --dataset acm_dblp --src acm --tar dblp \
  --methods a2gnn specreg dgsda sagda \
  --models plain plsa pairalign strurw \
  --seeds 0 1 2 3 4
```

**Full reproduction** uses `real-sweep`, which runs the paper's full
hyperparameter grid and selects per-(direction, backbone, model, seed) by
validation accuracy before reporting test accuracy:

```bash
# Table 1 -- ACM <-> DBLP
python run.py --device cuda real-sweep --dataset acm_dblp --src acm --tar dblp \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4
python run.py --device cuda real-sweep --dataset acm_dblp --src dblp --tar acm \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4

# Table 2 -- Twitch
python run.py --device cuda real-sweep --dataset twitch --src DE --tar ENGB \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4
python run.py --device cuda real-sweep --dataset twitch --src ENGB --tar DE \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4

# Table 3 -- CORA, L=6 and L=8
python run.py --device cuda real-sweep --dataset cora --k-labels 6 \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4
python run.py --device cuda real-sweep --dataset cora --k-labels 8 \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4

# Table 4 -- Airport
python run.py --device cuda real-sweep --dataset airport --src usa --tar brazil \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4
python run.py --device cuda real-sweep --dataset airport --src usa --tar europe \
  --methods a2gnn specreg dgsda sagda --models plain plsa pairalign strurw --seeds 0 1 2 3 4

# Table 5 -- calibration ablation
python run.py --device cuda csbm-calib --seeds 0 1 2 3 4 --methods plsa_cal plsa_uncal pairalign

# Figure 1 -- CSBM importance-weight estimation error
python run.py --device cuda csbm --mode fig1 --L 2 --n 5000 --seeds 0 1 2 3 4 --methods pairalign plsa

# Figure 2 -- estimation error vs. graph size
python run.py --device cuda csbm --mode fig2 --L 2 --seeds 0 1 2 3 4 --methods pairalign plsa
python run.py --device cuda csbm --mode fig2 --L 3 --seeds 0 1 2 3 4 --methods pairalign plsa
```

## License

MIT License, see `LICENSE`.