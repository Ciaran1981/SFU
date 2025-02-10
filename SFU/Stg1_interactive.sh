#!/bin/bash

while getopts ":m:c:d:h:" x; do
  case $x in
    h) 
      echo "Execute stage one of Soil for Unc process where environmental covariates are predicted from soil props - uses an interactive job so the process is known"
      echo "Usage: Stg1_interactive.sh -m model/directory -c my/csv.csv -d Out/directory"
      echo "-m MODEL     : modeldirectory/*.gz for prediction"
      echo "-c CSV       : input csv"
      echo "-d OUT       : output directory"        
      echo "-h	          : displays this message and exits."
      echo " "
      exit 0 
      ;;    
      	m)   
      MODEL=$OPTARG 
      ;;
      	c)   
      CSV=$OPTARG 
      ;;
        d)   
      OUT=$OPTARG 
      ;;     
    \?)
      echo "Stg1_interactive.sh: Invalid option: -$OPTARG" >&1
      exit 1
      ;;
    :)
      echo "Stg1_interactive.sh: Option -$OPTARG requires an argument." >&1
      exit 1
      ;;
  esac
done


# the cpus could be increased as it is multi thread, but the dataset is tiny so hardly worth it
srun --partition=short --cpus-per-task=1 --mem=1G --pty bash;

source activate SFU;

echo "executing SFU stage 1 interactive script"

# path also likely to change
python SFU/SFU_Pred_hpc.py -m ${MODEL} -c ${CSV}  -d ${OUT}






