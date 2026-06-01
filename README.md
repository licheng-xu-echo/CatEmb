# CatEmb
[![Ubuntu](https://img.shields.io/badge/Ubuntu-orange)](https://ubuntu.com/) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/) [![PyTorch 2.6.0+cu124](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red)](https://pytorch.org/) [![PyG 2.7.0](https://img.shields.io/badge/torch__geometric-2.7.0-green)](https://pytorch-geometric.readthedocs.io/) [![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-blueviolet)](https://www.rdkit.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of *"Stereoelectronic-aware catalyst embeddings from 2D graphs via 2D-3D multi-view alignment"*. The corresponding paper is under review. Preprint version is available at [ChemRxiv](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000279/v1).

CatEmb​ is a novel, stereoelectronic-aware molecular descriptor that generates compact, fixed-length embeddings directly from 2D molecular graphs (e.g. SMILES). It bridges the gap between readily accessible 2D structural inputs and the decisive 3D stereoelectronic properties essential for data-driven catalyst discovery.

![CatEmb_INTRO](img/TOC.png)

## ✨ Features
* **From 2D to 3D Properties:** Generates molecular embeddings that implicitly capture 3D geometric and energetic information using only 2D graphs as input (for user, it is just SMILES).
* **End-to-End Automation:** Eliminates the need for manual feature engineering, conformational searches, or expensive quantum-chemical calculations during inference.
* **Chemically Intuitive:** The learned embedding space provides a chemically meaningful similarity metric, effectively differentiating ligand classes based on subtle stereoelectronic variations.
* **Ready for Prediction:** Functions as a powerful molecular feature to enhance Quantitative Structure-Performance Relationship (QSPR) models for predicting catalytic outcomes.
* **Accelerates Discovery:** Enables efficient, similarity-based catalyst recommendation strategies for high-throughput experimentation and virtual screening.


## Installation
**1. Clone the repository**
```bash
git clone https://github.com/licheng-xu-echo/CatEmb.git
cd CatEmb
```

We recommend using `conda` to manage the environment.
```
# Create and activate a new conda environment
conda create -n catemb python=3.12 -y
conda activate catemb

# Install dependencies from requirements.txt
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.6.0+cu124.html --extra-index-url https://download.pytorch.org/whl/cu124
```

**2. Download pretrained model weights**

**Option 1:** Download from modelscope (recommended)
```bash
# there are 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 dimensional CatEmb models
cd catemb
modelscope download --model 'XuLiCheng2025/CatEmb' --include 'model_path/dim*' --local_dir '.'
cd ..
```

**Option 2:** Download the trained CatEmb model [weights](http://doi.org/10.6084/m9.figshare.31375579) (`dim[8~4096]LN.tar.gz`) and extract the archive. Place the extracted folder (model_path/) into the project's `catemb` directory.

```
CatEmb/
├── catemb/
│   ├── model_path/
│   │   └── dim8LN/          # Contains the trained model checkpoints
│   │   ├── dim16LN/          # Contains the trained model checkpoints
│   |   └── ... (other model folders)
│   ├── __init__.py
│   ├── data.py
│   └── ... (other source files)
├── requirements.txt
├── setup.py
└── ...
```
**3. Set up the environment and install**


```bash
# Install the catemb package
pip install .
```

## Basic usage

```python
from catemb import CatEmb
catemb_calc = CatEmb(device='cpu') # default dimension is 32
# catemb_calc = CatEmb(device='cpu', model_dim=4096) # specify the dimension.
cat_smi_lst = ['CN(C)c1ccc(P(C2CCCCC2)C2CCCCC2)cc1',
               'COc1ccc(OC)c(P(C2CCCCC2)C2CCCCC2)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C']
desc = catemb_calc.gen_desc(cat_smi_lst)
```

## Dataset download
The `CatCompDB`​ dataset used in this project is available for download via [Figshare](http://doi.org/10.6084/m9.figshare.31375579). Place the extracted folder (dataset/) and put it in the `CatEmb` folder.

```
CatEmb/
├── catemb/
├── dataset/
│   │   ├── processed/
│   │   └── rxn_data/
│   └── ...
├── requirements.txt
├── setup.py
└── ...
```

It consists of three key files, representing different stages of the dataset construction pipeline:
1. `original_smiles.npy`: The initial curated dataset containing 12,797​ molecules.
2. `lig_cat_dataset_new.npy`: The expanded dataset (66,664​ molecules) after algorithmically generating ligand-metal complexes.
3. `catcompdb.npy`: The final, refined dataset (62,755​ entries) containing structures optimized with xTB and filtered for convergence.
Additionally, there are some reaction dataset for benchmark.

## Train model
If you want to train your own model, you can use the following command:
```
python train.py --batch_size 4 --epoch 100 --tag test
```
More arguments and their instructions can be found in the `train.py` file.

## Notebooks
This repository includes several Jupyter notebooks in `notebook` folder that demonstrate key functionalities and analyses.

## Scripts
To run QSPR benchmark, you can use the following command:
```bash
conda run -n catemb python scripts/qsrp_morgan.py --dataset both
conda run -n rxnfp python scripts/qspr_rxnfp.py --dataset both
conda run -n 3DInfomax python scripts/qspr_3dinfomax.py --dataset both --mode both
conda run -n unimol python scripts/qspr_unimol2.py --dataset both --mode both
conda run -n catemb python scripts/qspr_catemb_all.py --dataset both --mode both
conda run -n catemb python scripts/qspr_morgan_catemb.py --dataset both --mode both
conda run -n catemb python scripts/qspr_rxnfp_catemb.py
```
Note: To run [unimol](https://github.com/deepmodeling/unimol_tools), [3dinfomax](https://github.com/HannesStark/3DInfomax), you need prepare the environment first.

To run 2D-3D alignment benchmark, you can use the following command:
```bash
conda run -n catemb python scripts/benchmark_embedding_dims.py
```

## Citation

Li-Cheng Xu, Fenglei Cao, Yuan Qi. Stereoelectronic-aware catalyst embeddings from 2D graphs via 2D-3D multi-view alignment. *ChemRxiv* **2026** DOI: 10.26434/chemrxiv.15000279/v1
```
@article{xu_2026_catemb,
    author = {Li-Cheng Xu  and Fenglei Cao  and Yuan Qi },
    title = {Stereoelectronic-aware catalyst embeddings from 2D graphs via 2D-3D multi-view alignment},
    journal = {ChemRxiv},
    volume = {2026},
    number = {0222},
    pages = {},
    year = {2026},
    doi = {10.26434/chemrxiv.15000279/v1},
    URL = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15000279/v1},
    eprint = {https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15000279/v1}
}
```