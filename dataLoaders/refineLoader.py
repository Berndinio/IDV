from __future__ import print_function, division

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from estimators.poseEstimator import FacePoseEstimator
from Utils import Utils

ffiles = []


class FEIPostDataset(Dataset):
    def __init__(self, root, sampleShape, maskType, transform=None):
        self.root_dir = root
        self.transform = transform
        self.sampleShape = sampleShape
        self.files = None
        self.maskType = maskType
        for root, dirs, files in os.walk(self.root_dir):
            self.files = files
            if ".keep" in self.files:
                self.files.remove(".keep")
            continue
        self.entities = list(set([f.split("-")[0] for f in self.files]))

        self.poseEstimator = FacePoseEstimator()

    def __len__(self):
        return len(self.files) - len(self.entities) * 4

    def __getitem__(self, idx):
        entityIdx = int(idx / 10)
        # source
        poseIdx = idx - entityIdx * 10 + 1
        entityIdx += 1
        #source
        faces = []
        while len(faces) <= 0:
            filePath = self.root_dir + "" + str(entityIdx) + "-" + str(poseIdx).zfill(2) + ".jpg"
            conditionImage = Image.open(filePath)
            conditionImage = transforms.Resize((self.sampleShape[1], self.sampleShape[2]), interpolation=2)(conditionImage)
            open_cv_image = np.array(conditionImage)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            faces, rects = self.poseEstimator.predict(open_cv_image)
            #just go one up, if you cant recognise the face of this one
            poseIdx = (poseIdx)%10 + 1
        embeddingCondition = self.poseEstimator.generatePoseEmbedding(faces[0], open_cv_image)


        # target
        faces = []
        while len(faces) <= 0:
            ran = int(random.random() * 10.0) + 1
            filePath = self.root_dir + "" + str(entityIdx) + "-" + str(ran).zfill(2) + ".jpg"
            targetImage = Image.open(filePath)
            targetImage = transforms.Resize((self.sampleShape[1],self.sampleShape[2]), interpolation=2)(targetImage)
            # heatmaps
            open_cv_image = np.array(targetImage)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            faces, rects = self.poseEstimator.predict(open_cv_image)
        if self.maskType == 0:
            mask = self.poseEstimator.generateMask(faces[0], open_cv_image)
        else:
            mask = self.poseEstimator.generateBboxMask(rects[0], open_cv_image)
        embeddingTarget = self.poseEstimator.generatePoseEmbedding(faces[0], open_cv_image)
        if self.transform:
            conditionImage = self.transform(conditionImage)
            targetImage = self.transform(targetImage)

        return conditionImage, targetImage, mask, torch.cat((embeddingCondition, embeddingTarget), 0)
        return conditionImage, targetImage, mask, embeddingTarget


if __name__ == "__main__":
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = FEIPostDataset(root="dataset/FEI/",
                             transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    conditionImage, targetImage, mask, embedding = dataset[1]
    if not mask is None:
        print(conditionImage.shape, targetImage.shape, mask.shape, embedding.shape)
