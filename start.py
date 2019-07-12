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




class Trainer:
    def __init__(self, sampleImage, numBlocks):
        self.ceLoss = torch.nn.BCELoss()
        self.sampleImage = sampleImage
        s = sampleImage.shape
        self.G1 = Generator(numBlocks, (s[0] + 68, s[1], s[2]), "G1", 1).to(Utils.g_device)
        self.G2 = Generator(numBlocks, (s[0] * 2, s[1], s[2]), "G2", 1).to(Utils.g_device)
        self.D = Discriminator((s[0] * 2, s[1], s[2])).to(Utils.g_device)

    def lossG1(self, I_b1, I_b, M_b):
        """
        :param M_b:     Pose mask
        :param I_b1:    Generated image (from G1)
        :param I_b:     Target image
        """
        loss = (I_b1 - I_b) * (1.0 + M_b)
        loss = torch.sum(torch.abs(loss))
        return loss

    def lossD(self, outD, realD):
        outD, realD = outD.float(), realD.float()
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
        loss += self.ceLoss(outD.float(), torch.ones(outD.shape).float())
        return loss

    def startTraining(self, numEpochs=10):

        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = FEIPostDataset(root="dataset/FEI/", sampleShape=self.sampleImage.shape,
                                 transform=data_transform)
        dataLoader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1, shuffle=False,
                                                 num_workers=1)
        print("Loaded Data")
        # stage 1
        optimizer = torch.optim.Adam(self.G1.parameters(), lr=0.00001, betas=(0.5, 0.999))
        for epoch in range(numEpochs+100):
            print("Running stage 1 training epoch " + str(epoch))
            running_loss = 0
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                conditionImages, targetImages, masks, embeddings = conditionImages.to(Utils.g_device), targetImages.to(Utils.g_device), masks.to(Utils.g_device), embeddings.to(Utils.g_device)
                print(str(i) + "/" + str(len(dataLoader)))
                inputs = torch.cat((conditionImages, embeddings), dim=1)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.G1(inputs)

                outputs = outputs[:, 34:37]
                print("Input", inputs.type(), inputs.min(), inputs.max(), inputs.shape)
                print("Target", targetImages.type(), targetImages.min(), targetImages.max(), targetImages.shape)
                print("Output", outputs.type(), outputs.data.min(), outputs.data.max(), outputs.shape)
                loss = self.lossG1(outputs, targetImages, masks)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 1 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    # save images
                    for x, img in enumerate(outputs):
                        toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                        toSave.save("results/1G-generated"+str(x)+".png")
                        toSave = transforms.ToPILImage(mode="RGB")(conditionImages[x].cpu())
                        toSave.save("results/1G-conditionImages"+str(x)+".png")
                        toSave = transforms.ToPILImage(mode="RGB")(targetImages[x].cpu())
                        toSave.save("results/1G-target"+str(x)+".png")
                continue

        # stage 2
        optimizer = torch.optim.Adam(list(self.G2.parameters()) + list(self.D.parameters()), lr=0.00001,
                                     betas=(0.5, 0.999))
        for epoch in range(numEpochs):
            print("Running stage 2 training epoch " + str(epoch))
            running_loss = 0
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                conditionImages, targetImages, masks, embeddings = conditionImages.to(Utils.g_device), targetImages.to(Utils.g_device), masks.to(Utils.g_device), embeddings.to(Utils.g_device)
                print(str(i) + "/" + str(len(dataLoader)))
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
                    targetDecision = torch.zeros((condition.shape[0], 1))
                else:
                    # generate real pair
                    generated = targetImages
                    condition = conditionImages
                    targetDecision = torch.ones((condition.shape[0], 1))

                discriminatorPair = torch.cat((generated, condition), dim=1)
                decision = self.D(discriminatorPair)
                # loss of D
                loss = self.lossD(decision, targetDecision)
                # loss of G2
                if targetDecision[0, 0] == 0:
                    loss = loss + self.lossG2(generated, targetImages, masks, decision, lamb=2)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    # save images
                    for x, img in enumerate(outputs):
                        print(img.min(), img.max())
                        toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                        toSave.save("results/2G-"+str(x)+".png")


if __name__ == "__main__":
    s = (3, 480, 640)
    factor = 2.0
    sample = np.zeros((int(s[0]), int(s[1] / factor), int(s[2] / factor)))
    Trainer(sample, 7).startTraining()
