########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataLoaders.refineLoader import FEIPostDataset


class Trainer:
    def __init__(self):
        self.ceLoss = torch.nn.BCELoss()

    def lossG1(self, I_b1, I_b, M_b):
        """
        :param M_b:     Pose mask
        :param I_b1:    Generated image (from G1)
        :param I_b:     Target image
        """
        loss = (I_b1-I_b) * (1.0+M_b)
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

    def startTraining(self):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.lossG1()
        dataset = FEIPostDataset(root="dataset/FEI/",
                                 transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=4, shuffle=True,
                                                     num_workers=4)
        conditionImage, targetImage, mask, embedding = dataset[1]
        if not mask is None:
            print(conditionImage.shape, targetImage.shape, mask.shape, embedding.shape)


if __name__ == "__main__":
    Trainer().startTraining()
