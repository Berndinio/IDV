# IDV
## Task
Facial Pose Manipulation transfered from https://arxiv.org/abs/1705.09368 ( Pose Guided Person Image Generation from Liqian Ma, Xu Jia, Qianru Sun, Bernt Schiele, Tinne Tuytelaars, Luc Van Gool)

## Abstract
Nowadays pose or shape manipulation is a largely applied field. Especially in CGI. This
report is about transfering the body pose manipulation method from Liqian Ma et al[1] to
face/head pose manipulation. They use a two staged training process. First they generate
a coarse pose manipulated image. Then they transfer details to the generated image by
using a GAN-like training setting. Additionally, masking is applied to mark regions of
interest.
Transfering this method to facial pose manipulation can be challenging by choosing the
pose representation, masking, upsampling method of the generators and implementing it
from scratch with the pytorch library.

## Additional material
Poster and Report.

## Install environment
Install needed modules in conda environment (python 3.5):

    conda install -c conda-forge opencv
    conda install pytorch-cpu torchvision-cpu -c pytorch
    conda install -c menpo dlib
    python -m pip install -U matplotlib
    conda install -c conda-forge scikit-image

or use the

    requirements.txt


Download the FEI dataset (https://fei.edu.br/~cet/facedatabase.html) and put it into <projectRoot>/dataset/FEI/
Download the face-landmark detector from https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2 and put it into <projectRoot>/estimators/pretrained_files

## Start training
Go into the root directory (where this README.md is).
Start the training with:

    python -m start

Starting at different training stages can be done with:

    python -m start -stage i -epochs n

Stages are:
- 0 = Training G1, G2, D
- 1 = Training G2, D. Loading G1 from file.
- 2 = Only generating images. Loading G1, G2, D from file.
- default = 0

According to this there will need to be the files:
- For stage 0 there do not need to be pretrained files.
- For stage 1, there needs to be a "G1.pt" in the results folder.
- For stage 2, there needs to be a "G1.pt" and "G2.pt" in the results folder.

Number of epochs is supplied by the epochs argument. Default = 30.
## Structure & Format & Recommendations
All trained files have the format:

    G1-<loss>-<epoch>.pt
    G2-<loss>-<epoch>.pt
    D-<loss>-<epoch>.pt
and are in the results/BBox_upsample/ folder. If you start at a stage>0, the files need to be renamed and moved. Best copy is:

    G1 with least loss
    G2 with highest epoch
    D with highest epoch

Generated images (stage==2) will lie in the created folder results/BBox_upsample/x-generated-fromG2-i.png (NOT "fromG1")

All generated images have the format:

- 1G-conditionImages<i>.png -- The condition image at stage 0. (Part of input for G1)
- 1G-generated<i>.png -- The generated image at stage 0. (Output of G1)
- 1G-target<i>.png -- The target image at stage 0. (Target output of G1)


- 2G-generated-diff-<i>.png -- The difference image of G2 at stage 1. (Output of G2)
- 2G-generated-<i>.png -- The final image of G2 at stage 1. (Output of G2 after adding it to G1 output)


- x-generated-diff-<i>.png -- The difference image of G2 at stage 2. (Output of G2)
- x-generated-fromG1-<i>.png -- The final image of G1 at stage 2. (Output of G1)
- x-generated-fromG2-<i>.png -- The final image of G2 at stage 2. (Output of G2 after adding it to G1 output)


There is also a csv file, which shows the SSIM and RGB L1 distance of 4 images between generated and target.
