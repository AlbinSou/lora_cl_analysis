# Code for the paper "An Empirical Analysis of Forgetting in Pre-trained Models with Incremental Low-Rank Updates"

## Installation

Clone this repository

```
git clone https://github.com/AlbinSou/lora_cl_analysis
```

Create an environment that uses python 3.10

```
conda create -n lora_cl python=3.10
conda activate lora_cl
```

Install Pytorch and Torchvision so that it uses your available cuda version (see [Pytorch website](https://pytorch.org/get-started/locally/) for more info)

```
pip install torch==2.0.0 torchvision==0.15
```

Install the remaining requirements

```
pip install -r requirements.txt
```

Setup environment variables so that project directory is recognized by python

```
conda env config vars set PYTHONPATH=/myhomedir/lora_cl_analysis
```

## Running the experiments

1) Create your deploy file where you will indicate the data folder and results folder in config/deploy
2) Run the lora_forget.py file in the experiments folder
3) Analyze the results with the provided notebooks

```
cd experiments

# Resnet experiments
python lora_forget.py deploy=my_deploy_file model=timresnet501k

# ViT experiments
python lora_forget.py deploy=my_deploy_file model=timvit1k
```
