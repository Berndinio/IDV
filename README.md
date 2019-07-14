# IDV
## Install environment
Install needed modules in conda environment (python 3.5):

    conda install -c conda-forge opencv
    conda install pytorch-cpu torchvision-cpu -c pytorch
    conda install -c menpo dlib
    python -m pip install -U matplotlib
    conda install -c conda-forge scikit-image

Go into the root directory (where this README.md is).

## Start training
Start the training with:

    python -m start

There are different modes for masks and upsampling methods.
The image size is variable (needs to stay the same in one training)<br />
==> You could use different datasets

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
Sometimes the process can be unstable and needs to be restarted.

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
