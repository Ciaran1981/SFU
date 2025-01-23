# SFU
Process an incoming file

## To install

conda env create -f sfu.yml

Alternatively, for a shorter wait (conda is quite slow these days), the mamba system is recommended, install this in your base conda then:

mamba env create -f sfu.yml

## To use

Either:

`Stg1_sub.sh -m model/directory -c my/csv.csv -d Out/directory`

`source/conda activate SFU`

`python SFU/SFU_Pred_hpc.py -m ${MODEL} -c ${CSV}  -d ${OUT}`


