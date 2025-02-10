# SFU
Process an incoming file

## To install

conda env create -f sfu.yml

Alternatively, for a shorter wait (conda is quite slow these days), the mamba system is recommended, install this in your base conda then:

mamba env create -f sfu.yml

## To use

One of the following.

A job submission:
```bash
SFU/Stg1_sub.sh -m model/directory -c my/csv.csv -d Out/directory
```

Starts an interactive job then finishes

```bash
SFU/Stg1_interactive.sh -m model/directory -c my/csv.csv -d Out/directory
```

Via python interactively

```bash
source/conda activate SFU

python SFU/SFU_Pred_hpc.py -m  model/directory -c my/csv.csv  -d Out/directory
```



