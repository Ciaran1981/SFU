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
from osgeo import gdal
from tqdm import tqdm
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
# test
#mdir= '/media/ciaran/464affee-823e-4d6b-a94b-babb27eb1894/SoilPropsPred/Chosen'
# csv='/home/ciaran/SFU/Test/TestData.csv'
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

"""
Starting with the 38 primary model outputs and a corresponding set of 38 normalised spatial covariate datasets: 

Loop through each spatial dataset. 

Within each dataset, loop through each 10m grid cell. 

Calculate the absolute difference A between the primary model value for that dataset, 
and the value in the 10m grid cell. 

Calculate P = 1-A. 

You now have 38 new spatial datasets. 

Create a new blank spatial dataset with the same size and resolution. 

Loop through every 10m grid cell. 

Loop through every one of the new spatial datasets. 

Calculate X=W x P0.5, where W is the R2 value for the corresponding model 
and P is the value in the spatial dataset. 

Add the 38 ‘X’ values together to get S. 

Calculate F=(S/T)2 where T is the sum of all 38 values of W. 

You now have a new spatial dataset made up of values of F. This is the dataset to be presented to the user. 
"""

#test
# we need the index of each band 
inras = '/media/ciaran/464affee-823e-4d6b-a94b-babb27eb1894/25kSTK/25ksmooth.vrt'
inras = '/media/ciaran/464affee-823e-4d6b-a94b-babb27eb1894/25kSTK/NC7550.tif'

outMap = '/media/ciaran/464affee-823e-4d6b-a94b-babb27eb1894/SoilPropsPred/TST/NCtst.tif'

# unfortunately there are some abrev in the input table so this is a corrected one
corr = ['Geology6','Temp10','Rain6','TopographicRoughnessIndex','Rain10','Rain9','Slope','Soil8',
         'Temp11','Rain7','Soil7','LC2','Soil9','Rain2','Rain4','Temp7','Rain8',
         'Rain5','Temp5','Temp6','Temp2','Aspect','Rain11','Temp1','ValleyDepth',
         'Soil10','Rain3','LC5','Geology8','MRVBF','Temp4','Temp9','MRRTF','Rain1',
         'Temp12','Rain12','Temp3','Temp8']

blist = list_bandnames(inras)

# get the inds and add 1 for band gdal
inds = [blist.index(c)+1 for c in corr]

# test
# r2csv = '/media/ciaran/464affee-823e-4d6b-a94b-babb27eb1894/SoilPropsPred/sfpred.csv'
r2df = pd.read_csv(r2csv)

# BTW aspect seems to be a negative r2 should dump that in future
r2sel = r2df[r2df['Var'].isin(ootnms)]

# set the index first
r2sel.set_index('Var', inplace=True)
# then use loc
r2sel = r2sel.loc[ootnms]
r2 = r2sel.r2.to_numpy()

r2 = np.expand_dims(r2, axis=0)

# this now needs to be in the model/band order so can apply to array in func below

# Think we can do something like the below blockwise and mis out creating the
# intermediate datasets

# Executing line at bottom
def _copy_dataset_config(inDataset, FMT='Gtiff', outMap='copy',
                         dtype=gdal.GDT_Int32, bands = 1):
    """Copies a dataset without the associated rasters.

    """

    
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    #dtype=gdal.GDT_Int32
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    # TODO bad but can't put None in options
    if FMT == 'MEM':
        outDataset = driver.Create(
            outMap, 
            x_pixels,
            y_pixels,
            bands,
            dtype)
    else:
        outDataset = driver.Create(
            outMap, 
            x_pixels,
            y_pixels,
            bands,
            dtype,
            options=['COMPRESS=LZW'])

    outDataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))
        
    outDataset.SetProjection(projection)
    
    return outDataset

def blocproc(block, cols, rows):
    
    """
    Make a block processing structure for a raster
    """
    
    # TODO - numpy solution
    coords = []
    for i in (range(0, rows, block)):
        if i + block < rows:
            numRows = block
        else:
            numRows = rows -i
            
        
        for j in range(0, cols, block):
            if j + block < cols:
                numCols = block
            else:
                numCols = cols - j
    
                
            coords.append((i, j, numRows, numCols))
            
    return coords


def list_bandnames(inras):
    
    """
    Return bandnames from a raster

    Parameters
    ----------
    
    inras: str
            input raster path
            
    Returns
    -------
    
    list of bandnames
    
    """
    bandlist = []
    rds = gdal.Open(inras)
    
    bands = np.arange(1, rds.RasterCount+1).tolist()
    for b in (bands):
        rb = rds.GetRasterBand(b)
        desc = rb.GetDescription()
        bandlist.append(desc)
    rds = None
    
    return bandlist

def proc_pixel_bloc(inras, outMap, inds, r2, blocksize=256, FMT=None,
                       dtype=gdal.GDT_Float32):
    """
    A block processing func 
    
    Parameters
    ------------------
        
    inras: string
                 path to image including the file fmt 'Myimage.tif'
    
    outMap: string
             path to output image excluding the file format 'pathto/mymap'
    
    inds: list
        band indices
    
    r2: np array
        array of r2 vals corresponding to models and input bands
    
    FMT: string
          optional parameter - gdal readable fmt
    
    blocksize: int 
                size of raster chunk in pixels 256 tends to be quickest
    
    dtype: int 
            a gdal dataype 

    """

    
    print('reading img')
    rds = gdal.Open(inras)
    
    outds = _copy_dataset_config(rds, outMap=outMap,
                                     dtype=dtype, bands=1)
    cols = int(rds.RasterXSize)
    rows = int(rds.RasterYSize)

    blp = blocproc(blocksize, cols, rows)
    
    #test with the vrt
    # i = 30000
    
    # j = 30000
    
    #vrrng = np.arange(1, 39)
    outBand = outds.GetRasterBand(1)
    
    for i, j, numRows, numCols in tqdm(blp):

        X = rds.ReadAsArray(j, i, xsize=numCols, ysize=numRows, band_list=inds)

        if X.max() == 0:
            continue              
        else:
            
            # To vectorize the subtraction below - do a bit of reshaping
            # can the subtraction be done without rshape? surely it can??
            X.shape = ((38,numRows*numCols))
            # # recall at this point the table is on its side
            # # now more shifting about 
            X = X.transpose() 
            X = np.where(np.isfinite(X),X,0)
            # relic but left in case
            X[X==-99999]=0 # saga no data
            X[X==-9999]=0 # again
            X[X==1e+20]=0 # climate nodata
            # the abs diff
            a = np.absolute(X - denorm)
            
            # TODO p**0.5 results in nans due to neg values
            # from 1-a. I think using the abs to get p is wrong, but not 
            # sure how to solve as yet
            
            # IN MATTS DOC - produces later nans due to python and neg numbers?
            #p = 1 - a
            # I guess here abs would eliminate the neg values gen'd above
            # which cause the nan problem later
            # Tho the meaning is lost from the 1- right?
            # NOT MATTS formula temp like this to get it to run
            p = np.absolute(1-a)
            
            # we don't bother creating new rasters as this can all be
            # done in mem
            # Need the R2 of every model
            xx = r2 * (p**0.5) # Matt has to power 0.5 produces nans
            # but if I do the figures manually its fine 
            #test 
            # pp = p[0,:]
            #  pp**0.5 nans
            # pp[1]**0.5 nan
            # and yet this works - so it is np array related??
            # problem is seemingly that input data is negative - but this is
            # inevitable
            # -0.7382821359340888 ** 0.5
            
            # Add the 38 ‘X’ values together to get S.
            # so in this case we are summing each row
            # nans wtf
            sm = xx.sum(axis=1)
            
            # sum of r2s
            t  = r2.sum()
            
            f = (sm / t)**2
            
            f.shape = (numRows, numCols)
            
            # predictClass = model.predict(X)
            
            # predictClass = np.reshape(predictClass, (numRows, numCols))
                         
            
            outBand.WriteArray(f,j,i)

    print('flushing cache')
    outds.FlushCache()
    outds = None   



# dae it
print('producing raster')
proc_pixel_bloc(inras, outMap, inds, r2, blocksize=256, 
                       dtype=gdal.GDT_Float32)
print('raster written')