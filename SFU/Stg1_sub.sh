#!/bin/bash

#SBATCH --mem=1G

#SBATCH --array=0-231

#SBATCH --error

#SBATCH --mail-user=ciaran.robb@hutton.ac.uk
#SBATCH --mail-type==BEGIN,END,FAIL,ALL

while getopts ":i:o:m:c:r:d:h:" x; do
  case $x in
    h) 
      echo "Execute stage one of Soil for Unc process where environmental covariates are predicted from soil props"
      echo "Usage: Stg1_sub.sh -i my/covariates -o my/outfiles  -m model/directory -c my/csv.csv -r my/r2csv.csv -d Out/directory"
      echo "-i INDIR     : input covariate dir - contains tiled raster stacks of covariates"
      echo "-o OUTDIR    : output directory for raster tiles"
      echo "-m MODEL     : modeldirectory/*.gz for prediction"
      echo "-c CSV       : input csv"
      echo "-r R2CSV     : input r2 csv containing the column headers Var and r2"
      echo "-d OUT       : output directory"        
      echo "-h	          : displays this message and exits."
      echo " "
      exit 0
      ;;    
      	i)   
      INDIR=$OPTARG  
      ;;
      	o)   
      OUTDIR=$OPTARG  
      ;;      
      	m)   
      MODEL=$OPTARG 
      ;;
      	c)   
      CSV=$OPTARG 
      ;;
        r)   
      R2CSV=$OPTARG 
      ;;
        d)   
      OUT=$OPTARG 
      ;;     
    \?)
      echo "Stg1_sub.sh: Invalid option: -$OPTARG" >&1
      exit 1
      ;;
    :)
      echo "Stg1_sub.sh: Option -$OPTARG requires an argument." >&1
      exit 1
      ;;
  esac
done


# env - likely to change
source activate SFU;

echo "executing SFU stage 1 script"

FILES=(${INDIR}/*.tif);

# path also likely to change
echo "executing python script"
python SFU/SFU_Pred_hpc.py -i ${FILES[$SLURM_ARRAY_TASK_ID]}  -o ${OUTDIR} -m ${MODEL} -c ${CSV} -r2 ${R2CSV} -d ${OUT}






