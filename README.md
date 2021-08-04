# Unsupervised semantic Segmentation with Pose Prior (USPP)
By: Max Midwinter </br>
CVISS Labs, Dept. CEE, University of Waterloo <br/>
Rogers 5G Smart Infrastructure, 2021

## Demo
Run main.py for sample program

## Own Data

1. Call prepDefect ( ) in main.py <br/>
prepDefect takes 4 parameters:
   * REF_DIR: dir containing the reference frame 
   * SUB_DIR: dir containing other images of the defect
   * AUG_DIR: dir where the preprocessed images are saved
   * scale: % to resize the raw images (Depends on your GPU and number of filters)
     * Keep resolution below 500x500 

2. Call scribbsDefect ( ) in main.py <br/>
scribbsDefect takes 2 parameters:
    * AUG_DIR: dir where images are saved
    * scribbs: scribbles to feed in
      * usually leave at None
    * save: save output image
      * For Debug save image (saved in current directory)

## Details


