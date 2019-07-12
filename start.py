########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np

from dataLoaders.refineLoader import FEIPostDataset
from nets.refineGenerator import Generator, Discriminator

from Utils import Utils
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, sampleImage, numBlocks, upsampleType=1):
        self.ceLoss = torch.nn.BCELoss()
        self.sampleImage = sampleImage
        s = sampleImage.shape
        self.upsampleType = upsampleType
        # flag = G1 or G2
        # G1 = With FC in mid
        # G2 = With Conv in mid
        # mode = 1 or 2
        # 1 = With Upsampling (nearest neighbor)
        # 2 = With Deconvolution
        self.G1 = Generator(numBlocks, (s[0] + 68, s[1], s[2]), "G1", upsampleType).to(Utils.g_device)
        self.G2 = Generator(numBlocks, (s[0] * 2, s[1], s[2]), "G2", upsampleType).to(Utils.g_device)
        self.D = Discriminator((s[0] * 2, s[1], s[2])).to(Utils.g_device)

    def lossG1(self, I_b1, I_b, M_b):
        """
        :param M_b:     Pose mask
        :param I_b1:    Generated image (from G1)
        :param I_b:     Target image
        """
        loss = (I_b1 - I_b) * (0.5 + M_b)
        loss = torch.sum(torch.abs(loss))
        return loss

    def lossD(self, outD, realD):
        outD, realD = outD.float().to(Utils.g_device), realD.float().to(Utils.g_device)
        return self.ceLoss(outD, realD)

    def lossG2(self, I_b2, I_b, M_b, outD, lamb=1):
        """
        :param outD:    Prediction from discriminator
        :param lamb:    Hyperparameter
        :param M_b:     Pose mask
        :param I_b2:    Generated image (from G2)
        :param I_b:     Target image
        """
        loss = lamb * self.lossG1(I_b2, I_b, M_b)
        loss += self.ceLoss(outD.float().to(Utils.g_device), torch.ones(outD.shape).float().to(Utils.g_device))
        return loss

    def startTraining(self, numEpochs=10, maskType=0):
        """
        :param maskType:    0=Landmark, 1=BBox
        """

        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = FEIPostDataset(root="dataset/FEI/", sampleShape=self.sampleImage.shape,
                                 maskType=maskType, transform=data_transform)
        dataLoader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4, shuffle=False,
                                                 num_workers=4)
        print("Loaded Data")
        # stage 1
        optimizer = torch.optim.Adam(self.G1.parameters(), lr=0.00001, betas=(0.5, 0.999))
        epochLosses = []
        for epoch in range(numEpochs):
            print("Running stage 1 training epoch " + str(epoch)+"/"+str(numEpochs))
            allLoss = 0
            #path of plots
            mType = "BBox"
            if maskType == 0:
                mType = "keypoints"
            up = "deconv"
            if self.upsampleType == 1:
                up = "upsample"
            folderPath = "results/"+mType+"_"+up+"/"
            print(folderPath)
            import os
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            #begin training
            running_loss = 0
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                conditionImages, targetImages, masks, embeddings = conditionImages.to(Utils.g_device), targetImages.to(Utils.g_device), masks.to(Utils.g_device), embeddings.to(Utils.g_device)
                inputs = torch.cat((conditionImages, embeddings), dim=1)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.G1(inputs)

                outputs = outputs[:, :3]
                loss = self.lossG1(outputs, targetImages, masks)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                allLoss += loss.item()

                if i==0:
                    running_loss = 0
                if i % 20 == 0 and not i == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    # save images
                    for x, img in enumerate(outputs):
                        toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                        toSave.save(folderPath+"1G-generated"+str(x)+".png")
                        toSave = transforms.ToPILImage(mode="RGB")(conditionImages[x].cpu())
                        toSave.save(folderPath+"1G-conditionImages"+str(x)+".png")
                        toSave = transforms.ToPILImage(mode="RGB")(targetImages[x].cpu())
                        toSave.save(folderPath+"1G-target"+str(x)+".png")
            epochLosses.append(allLoss/(len(dataLoader) * self.sampleImage.shape[0] * self.sampleImage.shape[1] * self.sampleImage.shape[2]))
            plt.plot(epochLosses)
            plt.title(mType+" "+up)
            plt.savefig(folderPath+"epochLoss.png")
            plt.clf()
            if epoch%(numEpochs/10)==0 and not epoch==0:
                torch.save(self.G1, folderPath+"G1-"+("%.3f"%epochLosses[-1])+"-"+str(epoch)+".pt")
        torch.save(self.G1, folderPath+"G1-"+("%.3f"%epochLosses[-1])+"-"+str(numEpochs)+".pt")

        # stage 2
        optimizer = torch.optim.Adam(list(self.G2.parameters()) + list(self.D.parameters()), lr=0.00001,
                                     betas=(0.5, 0.999))
        for epoch in range(0):
            print("Running stage 2 training epoch " + str(epoch)+"/"+str(numEpochs))
            running_loss = 0
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                conditionImages, targetImages, masks, embeddings = conditionImages.to(Utils.g_device), targetImages.to(Utils.g_device), masks.to(Utils.g_device), embeddings.to(Utils.g_device)
                ran = random.random()
                if ran >= 0.5:
                    inputs = torch.cat((conditionImages, embeddings), dim=1)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward things
                    outputs = self.G1(inputs)
                    outputs = outputs[:, :3]
                    inputs2 = torch.cat((conditionImages, outputs), dim=1)
                    outputs2 = self.G2(inputs2)
                    outputs2 = outputs2[:, 1:4]
                    outputs3 = outputs2 + outputs
                    # generate fake Pair
                    generated = outputs3
                    condition = conditionImages
                    targetDecision = torch.zeros((condition.shape[0], 1)).to(Utils.g_device)
                else:
                    # generate real pair
                    generated = targetImages
                    condition = conditionImages
                    targetDecision = torch.ones((condition.shape[0], 1)).to(Utils.g_device)

                discriminatorPair = torch.cat((generated, condition), dim=1)
                decision = self.D(discriminatorPair)
                # loss of D
                loss = self.lossD(decision, targetDecision).to(Utils.g_device)
                # loss of G2
                if targetDecision[0, 0] == 0:
                    loss = loss + self.lossG2(generated, targetImages, masks, decision, lamb=2)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 0 and not i == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    # save images
                    for x, img in enumerate(outputs):
                        toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                        toSave.save("results/2G-"+str(x)+".png")


if __name__ == "__main__":
    s = (3, 480, 640)
    factor = 4.0
    sample = np.zeros((int(s[0]), int(s[1] / factor), int(s[2] / factor)))

    # upsampleType = 1 or 2
    # 1 = With Upsampling (nearest neighbor)
    # 2 = With Deconvolution
    # maskType = 0 or 1
    # 0 = keypoints
    # 1 = BBox
    #=============================>
    for up in [1]:
        for mask in [1]:
            Trainer(sample, 6, up).startTraining(500, mask)
