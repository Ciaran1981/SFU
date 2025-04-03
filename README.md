# SFU
Process an incoming file

## To install

```bash
conda env create -f sfu.yml
```
Alternatively, for a shorter wait (conda is quite slow these days), the mamba system is recommended, install this in your base conda then:

```bash
mamba env create -f sfu.yml
```
## To use

```bash
chmod +x SFU/*.py
chmod +x SFU/*.sh
```

One of the following.

A job submission:
```bash
sbatch SFU/Stg1_sub.sh -i input/raster/directory -o output/raster/directory -m model/directory -c Test/TestData.csv -r Test/sfpred.csv -d Out/directory
```


Via python interactively

```bash
source/conda activate SFU

python SFU/SFU_Pred_hpc.py -i input/raster.tif -o output/raster/directory -m  model/directory -c Test/TestData.csv -r2 Test/sfpred.csv  -d Out/directory
```



