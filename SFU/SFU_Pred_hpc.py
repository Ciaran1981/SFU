#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ciaran robb
"""
import numpy as np
import os
from glob import glob
from joblib import Parallel, delayed, load
import shutil as sh
import pandas as pd
import argparse

# Get input parameters

csvhelp = ('The input csv with a header of '
           'ColR,ColG,ColB,LOI,Sand,Silt,Clay,Ex_Ca,Ex_Mg,Ex_Na,Ex_K,Ex_H,'
           'PHH2O,PHCaCl2,C,N,Depth')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modeldir", type=str, required=True, 
                    help="Input model directory containing .gz files")

parser.add_argument("-c", "--csv", type=str, required=True, 
                    help=csvhelp)

parser.add_argument("-d", "--outdir", type=str, required=True, 
                    help="The outputdir for prediction csv")

parser.add_argument("-t", "--threads", type=int, 
                    required=False, default=False,
                    help="No of threads to use - likely a pointless option")

parser.add_argument('-para','--parallel',  type=bool, 
                    help='process in parallel with joblib, likely a pointless option',
                    required=False, default=False)

args = parser.parse_args() 


# read in the model list
mdir = args.modeldir

#  (currently sklearn 1.3))
# projects/jhi/soils/202411_SoilForUNC/models

modelist = glob(os.path.join(mdir, '*.gz'))

# the input var names for ref
# varnms = ['ColR','ColG','ColB','LOI','Sand','Silt','Clay',
#           'Ex_Ca','Ex_Mg','Ex_Na','Ex_K',
#           'Ex_H','PHH2O','PHCaCl2','C','N','Depth']

# Normalisation was applied when the models trained - this will eventually change
# likely within pipeline so will negate this step
xmin = np.array([48,37,12,0.31,2.70433333333333,0,0,0,0,0,0,0,3.285,2.4,0,0,2])
xmax = np.array([250,224,215,28.1834679133021,100,85.68,64,141.1,25.82,
                 14.23,4.15,136,9.02,8.16,18.42432455,4.68,100])

csv = args.csv

df = pd.read_csv(csv)

x = df.loc[0].to_numpy()

# the min/max normaliser formula (matches the norm's sheet line 1 as expected)
norm = (x - xmin) / (xmax - xmin)

norm = np.expand_dims(norm, axis=0)

def mdl_pred(model, norm):
    
    mdl = load(model)
    pred = mdl.predict(norm)
    return pred[0]

ootnms = [os.path.basename(m)[:-11] for m in modelist]

# For such a tiny task para seems pointless waste on HPC
if args.parallel:
    print('joblib parallel')
    predlist = Parallel(n_jobs=args.threads,
                        verbose=1)(delayed(mdl_pred)(m,
                                                     norm) for m in (modelist))
    # 1.5secs...                                         
else:                                                    
    print('single proc')
    # in sequence
    predlist = [mdl_pred(m, norm) for m in modelist]
    # quicker than para 0.8 seconds   

predarr = np.array(predlist)

predarr.shape = (1, 38)

# Denorm the prediction - hard coded due to pre-norm'd training extending
# to the pred vals
xmx = np.array([  1,  11.02464485, 224.80300903,  23.92248726,
                 423.4359436 , 328.13192749,  42.68635559,   1,
                 8.51279354, 287.03897095,   1,   1,
                 1, 405.50778198, 221.22109985,  15.85888958,
                 332.54525757, 213.98010254,  11.49597836,  14.11598778,
                 6.05841732, 359.84921265, 404.66717529,   6.48538971,
                 564.09185791,   1, 340.47283936,   1,
                 1,   7.2966938 ,   8.68830872,  13.50822639,
                 7.2966938 , 464.06140137,   6.67719173, 442.91113281,
                 6.93636751,  15.52760124])

xmn = np.array([ 0,  0,  2.42623711,  0,
                    4.86868000,  3.87871075,  0,  0,
                    0,  2.79953647,  0,  0,
                    0,  4.28008080,  2.79333138,  0,
                    3.25917792,  2.54678655,  0,  0,
                   -2.35430264, -1.00000000,  4.66254425, -2.11750484,
                    0,  0,  4.01164675,  0,
                    0,  6.15285600e-13,  0,  0,
                    6.15285600e-13,  5.38492775, -1.95360851,  5.10766506,
                   -1.34422207,  0])

denorm = predarr * (xmx - xmn) + xmn

# output                                      
oot = pd.DataFrame(denorm, columns=ootnms)

# Assume incoming file has unique ident

hd, tl = os.path.split(csv[:-4]+'_stg1_pred.csv')

ootpth = os.path.join(args.outdir, tl)
oot.to_csv(ootpth)
print('file saved')










