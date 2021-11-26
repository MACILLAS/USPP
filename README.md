# Unsupervised semantic Segmentation with Pose Prior (USP)
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

## Docker Serving

### Install Docker
```
pip install docker
```

### Pull Compatible Tensorflow Docker Image
```
# Let me know if this does not work...
docker pull tensorflow/tensorflow:2.4.2-gpu
```

### Build Docker Image
Build docker image usp with your choice of tag
```
docker docker build -t usp:TAG .
```

### Run Docker Image
```angular2html
docker run usp:TAG
```
If you are running on local computer this command will start a dev server.
(i.e. http://172.17.0.2:5000/)

You can now push your docker image to a container registry of your choice and deploy a kubernetes service...

To take advantage of parallel inference... Run main_docker.py (with your API)



