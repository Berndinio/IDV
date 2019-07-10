########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

from dataLoaders.refineLoader import FEIPostDataset
from nets.refineGenerator import Generator, Discriminator


class Trainer:
    def __init__(self, sampleImage):
        self.ceLoss = torch.nn.BCELoss()
        self.G1 = Generator(6, sampleImage.shape, "G1", 2)
        self.G2 = Generator(6, sampleImage.shape, "G2", 2)
        self.D = Discriminator(sampleImage.shape)

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
        loss += self.ceLoss(outD, 1)
        return loss

    def startTraining(self, numEpochs=10):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.lossG1()
        dataset = FEIPostDataset(root="dataset/FEI/",
                                 transform=data_transform)
        dataLoader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
        #stage 1
        optimizer = torch.optim.Adam(self.G1.parameters(), lr=0.00001, betas=(0.5, 0.999))
        for epoch in range(numEpochs):
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                inputs = torch.cat((conditionImages, embeddings), dim=1)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.G1(inputs)
                loss = self.lossG1(outputs, targetImages, masks)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        #stage 2
        optimizer = torch.optim.Adam((self.G2.parameters(), self.D.parameters()), lr=0.00001, betas=(0.5, 0.999))
        for epoch in range(numEpochs):
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                ran = random.random()
                if ran >= 0.5:
                    inputs = torch.cat((conditionImages, embeddings), dim=1)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward things
                    outputs = self.G1(inputs)
                    inputs2 = torch.cat((conditionImages, outputs), dim=1)
                    outputs2 = self.G2(inputs2)
                    outputs3 = outputs2 + outputs
                    #generate fake Pair
                    generated = outputs3
                    condition = conditionImages
                    targetDecision = 0
                else:
                    #generate real pair
                    generated = targetImages
                    condition = conditionImages
                    targetDecision = 1

                discriminatorPair = torch.cat((generated, condition), dim=1)
                decision = self.D(discriminatorPair)
                #loss of D
                loss = self.lossD(decision, targetDecision)
                #loss of G2
                if targetDecision == 0:
                    loss += self.lossG2(generated, targetImages, masks, decision, lamb=2)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
if __name__ == "__main__":
    Trainer().startTraining()
