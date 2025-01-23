#!/bin/bash

#SBATCH --mem=1G

#SBATCH --error

#SBATCH --mail-user=ciaran.robb@hutton.ac.uk
#SBATCH --mail-type==BEGIN,END,FAIL,ALL

while getopts ":m:c:d:h:" x; do
  case $x in
    h) 
      echo "Execute stage one of Soil for Unc process where environmental covariates are predicted from soil props"
      echo "Usage: Stg1_sub.sh -m model/directory -c my/csv.csv -d Out/directory"
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

# path also likely to change
python SFU/SFU_Pred_hpc.py -m ${MODEL} -c ${CSV}  -d ${OUT}






