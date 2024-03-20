# CXIL: Continous Explainable Interactive Machine Learning 

This repository contains the code to out paper "CXIL: Continous Explainable Interactive Machine Learning"-

## Installation 

Clone this repository and install the repository with pip.
```
pip install .
```

## Usage 

## Rerun the Experiments
To replicate the paper results, please follow the three steps below. 

### Generate the Data 

The script to generate the (semi-) synthtic data can be found in ./CXIL/data.
Please run the scripts before running the experiments. 

### Run the Experiments

Run the bash Scripts: 
```
sh 1_Finetuning.sh
sh 2_Sanity.sh 
sh 2_RRR.sh
sh 2_Caipi.sh
```
### Evaluate 

After running the experiments the tables can be reproduces by: 
```
python A_Plots.py
python B_Plots_v2.py
```