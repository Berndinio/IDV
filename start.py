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
from skimage.measure import compare_ssim as ssim

from Utils import Utils
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, sampleImage, numBlocks, upsampleType=1, maskType=1):
        s = sampleImage.shape

        # dataset
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataset = FEIPostDataset(root="dataset/FEI/", sampleShape=s,
                                 maskType=maskType, transform=data_transform)
        conditionImage, targetImage, mask, embeddings = self.dataset[0]
        print("Dataset shapes: ", conditionImage.shape, targetImage.shape, mask.shape, embeddings.shape)
        #some variables
        self.ceLoss = torch.nn.BCELoss()
        self.sampleImage = conditionImage
        self.upsampleType = upsampleType
        self.maskType = maskType
        # maskType = 0 or 1
        # 0 = keypoints
        # 1 = BBox
        # flag = G1 or G2
        # G1 = With FC in mid
        # G2 = With Conv in mid
        # mode = 1 or 2
        # 1 = With Upsampling (nearest neighbor)
        # 2 = With Deconvolution
        #######Generator(N, imageSize=(3, 100, 100), flag="G1", mode=2, linearScaling=1, residual=True)
        self.G1 = Generator(numBlocks, (s[0] + embeddings.shape[0], s[1], s[2]), "G1", upsampleType, 16, True).to(Utils.g_device)
        self.G2 = Generator(numBlocks-2, (s[0] * 2, s[1], s[2]), "G2", upsampleType, 3, True).to(Utils.g_device)
        self.cutOff = 4
        self.D = Discriminator((s[0] * 2, s[1]-self.cutOff, s[2]-self.cutOff)).to(Utils.g_device)

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
        loss = loss + self.ceLoss(outD.float().to(Utils.g_device), torch.ones(outD.shape).float().to(Utils.g_device))
        return loss

    def startTraining(self, numEpochs=10, trainingStage=0, loaderPrefix="factor4_upsampling_bbox"):
        """
        :param maskType:    0=Landmark, 1=BBox
        """
        dataLoader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
        print("Loaded Data")

        #path of plots
        mType = "BBox"
        if self.maskType == 0:
            mType = "keypoints"
        up = "deconv"
        if self.upsampleType == 1:
            up = "upsample"
        folderPath = "results/"+mType+"_"+up+"/"
        print(folderPath)

        # stage 0
        if(trainingStage <= 0):
            print("Training G1")
            optimizer = torch.optim.Adam(self.G1.parameters(), lr=0.00001, betas=(0.5, 0.999))
            epochLosses = []
            for epoch in range(numEpochs):
                print("Running stage 1 training epoch " + str(epoch)+"/"+str(numEpochs))
                allLossInEpoch = 0
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

                    loss = self.lossG1(outputs, targetImages, masks)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    allLossInEpoch += loss.item()

                    if i==0:
                        running_loss = 0
                    if i % 20 == 0 and not i == 0:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, (i + 1)/len(dataLoader)*100.0, running_loss / 2000))
                        running_loss = 0.0
                        # save images
                        for x, img in enumerate(outputs):
                            toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                            toSave.save(folderPath+"1G-generated"+str(x)+".png")
                            toSave = transforms.ToPILImage(mode="RGB")(conditionImages[x].cpu())
                            toSave.save(folderPath+"1G-conditionImages"+str(x)+".png")
                            toSave = transforms.ToPILImage(mode="RGB")(targetImages[x].cpu())
                            toSave.save(folderPath+"1G-target"+str(x)+".png")
                epochLosses.append(allLossInEpoch/(len(dataLoader) * self.sampleImage.shape[0] * self.sampleImage.shape[1] * self.sampleImage.shape[2]))
                plt.plot(epochLosses)
                plt.title(mType+" "+up)
                plt.savefig(folderPath+"epochLossG1.png")
                plt.clf()
                if epoch%(numEpochs/10)==0 and not epoch==0:
                    torch.save(self.G1, folderPath+"G1-"+("%.3f"%epochLosses[-1])+"-"+str(epoch)+".pt")
            torch.save(self.G1, folderPath+"G1-"+("%.3f"%epochLosses[-1])+"-"+str(numEpochs)+".pt")
        else:
            print("Loading G1", "results/"+loaderPrefix+"G1.pt")
            self.G1 = torch.load("results/"+loaderPrefix+"G1.pt")
        # stage 1
        if(trainingStage <= 1):
            optimizer = torch.optim.Adam(list(self.G2.parameters()) + list(self.D.parameters()), lr=0.000002,
                                         betas=(0.5, 0.999))
            epochLosses = []
            lamb = 1.0
            for epoch in range(numEpochs):
                print("Running stage 2 training epoch " + str(epoch)+"/"+str(numEpochs))
                running_loss = 0
                allLossInEpoch = 0
                import os
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
                for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                    conditionImages, targetImages, masks, embeddings = conditionImages.to(Utils.g_device), targetImages.to(Utils.g_device), masks.to(Utils.g_device), embeddings.to(Utils.g_device)
                    ran = random.random()
                    optimizer.zero_grad()

                    if ran >= 0.5 and epoch>=1:
                        inputs = torch.cat((conditionImages, embeddings), dim=1)
                        # zero the parameter gradients

                        # forward things
                        outputs = self.G1(inputs)
                        inputs2 = torch.cat((conditionImages, outputs), dim=1)
                        outputs2 = self.G2(inputs2)
                        outputs3 = outputs2 + outputs
                        # generate fake Pair
                        generated = outputs3.clamp(0.0, 1.0)
                        condition = conditionImages
                        targetDecision = torch.zeros((condition.shape[0])).to(Utils.g_device)
                    else:
                        # generate real pair
                        generated = targetImages
                        condition = conditionImages
                        targetDecision = torch.ones((condition.shape[0])).to(Utils.g_device)

                    if(self.cutOff == 0):
                        discriminatorPair = torch.cat((
                                generated,
                                condition), dim=1)
                    else:
                        discriminatorPair = torch.cat((
                                generated[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff],
                                condition[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]), dim=1)
                    decision = self.D(discriminatorPair)
                    # loss of D
                    loss = self.lossD(decision, targetDecision).to(Utils.g_device)
                    # loss of G2
                    if targetDecision[0] == 0:
                        #lamb *= 0.999
                        loss += self.lossG2(generated, targetImages, masks, decision, lamb=lamb)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    allLossInEpoch += loss.item()
                    if i % 20 == 0 and not i == 0:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, (i + 1)/len(dataLoader)*100.0, running_loss / 2000))
                        running_loss = 0.0
                        # save images
                        for x, img in enumerate(generated):
                            toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                            toSave.save(folderPath+"2G-generated-fromG2-"+str(x)+".png")
                            if targetDecision[0] == 0:
                                toSave = transforms.ToPILImage(mode="RGB")(outputs2[x].cpu())
                                toSave.save(folderPath+"2G-generated-diff-"+str(x)+".png")
                epochLosses.append(allLossInEpoch/(len(dataLoader) * self.sampleImage.shape[0] * self.sampleImage.shape[1] * self.sampleImage.shape[2]))
                plt.plot(epochLosses)
                plt.title(mType+" "+up)
                plt.savefig(folderPath+"epochLossG2.png")
                plt.clf()
                if epoch%(numEpochs/10)==0 and not epoch==0:
                    torch.save(self.G2, folderPath+"G2-"+("%.3f"%epochLosses[-1])+"-"+str(epoch)+".pt")
                    torch.save(self.D, folderPath+"D-"+("%.3f"%epochLosses[-1])+"-"+str(epoch)+".pt")
            torch.save(self.G2, folderPath+"G2-"+("%.3f"%epochLosses[-1])+"-"+str(numEpochs)+".pt")
            torch.save(self.D, folderPath+"D-"+("%.3f"%epochLosses[-1])+"-"+str(epoch)+".pt")
        else:
            print("Loading G2 and D", "results/"+loaderPrefix+"G2.pt", "results/"+loaderPrefix+"D.pt")
            self.G2 = torch.load("results/"+loaderPrefix+"G2.pt")
            #self.D = torch.load("results/"+loaderPrefix+"D.pt")
        # stage 2 ==> just generate some
        if(trainingStage <= 2):
            print("Running stage 3. Generating...")
            import os
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            finalString = "SSIM RGB-Loss\n"
            for i, (conditionImages, targetImages, masks, embeddings) in enumerate(dataLoader, 0):
                conditionImages, targetImages, masks, embeddings = conditionImages.to(Utils.g_device), targetImages.to(Utils.g_device), masks.to(Utils.g_device), embeddings.to(Utils.g_device)
                inputs = torch.cat((conditionImages, embeddings), dim=1)

                # forward things
                outputs = self.G1(inputs)
                inputs2 = torch.cat((conditionImages, outputs), dim=1)
                outputs2 = self.G2(inputs2)
                generated = outputs2 + outputs
                generated = generated.clamp(0.0, 1.0)
                if not (self.cutOff == 0):
                    outputs = outputs[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]
                    inputs2 = inputs2[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]
                    outputs2 = outputs2[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]
                    generated = generated[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]
                    conditionImages = conditionImages[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]
                    targetImages = targetImages[:, :, self.cutOff:-self.cutOff, self.cutOff:-self.cutOff]


                for x, img in enumerate(generated):
                    toSave = transforms.ToPILImage(mode="RGB")(img.cpu())
                    toSave.save(folderPath+"x-generated-fromG2-"+str(x)+".png")
                    toSave = transforms.ToPILImage(mode="RGB")(outputs[x].cpu())
                    toSave.save(folderPath+"x-generated-fromG1-"+str(x)+".png")
                    toSave = transforms.ToPILImage(mode="RGB")(conditionImages[x].cpu())
                    toSave.save(folderPath+"x-generated-conditionImages-"+str(x)+".png")
                    toSave = transforms.ToPILImage(mode="RGB")(targetImages[x].cpu())
                    toSave.save(folderPath+"x-generated-target-"+str(x)+".png")
                    toSave = transforms.ToPILImage(mode="RGB")(outputs2[x].cpu())
                    toSave.save(folderPath+"x-generated-diff-"+str(x)+".png")

                    sourceImg, generatedImg1, generatedImg2, targetImage = \
                                torch.transpose(conditionImages[x].cpu(), 0, 1), \
                                torch.transpose(outputs[x].cpu(), 0, 1), \
                                torch.transpose(img.cpu(), 0, 1), \
                                torch.transpose(targetImages[x].cpu(), 0, 1)
                    sourceImg, generatedImg1, generatedImg2, targetImage = \
                                torch.transpose(sourceImg, 1, 2), \
                                torch.transpose(generatedImg1, 1, 2), \
                                torch.transpose(generatedImg2, 1, 2), \
                                torch.transpose(targetImage, 1, 2)
                    sourceImg, generatedImg1, generatedImg2, targetImage = \
                                sourceImg.detach().numpy(), \
                                generatedImg1.detach().numpy(), \
                                generatedImg2.detach().numpy(), \
                                targetImage.detach().numpy()

                    finalString += str(ssim(sourceImg, targetImage, data_range=1.0, multichannel=True))+ ","
                    finalString += str(ssim(generatedImg1, targetImage, data_range=1.0, multichannel=True))+ ","
                    finalString += str(ssim(generatedImg2, targetImage, data_range=1.0, multichannel=True))+ ","
                    finalString += str(np.sum(np.abs(generatedImg2 - targetImage))/(generatedImg2.shape[0]*generatedImg2.shape[1]*generatedImg2.shape[2])) + "\n"
                if i==2:
                    f = open(folderPath+"results.txt", "w")
                    f.write(finalString)
                    f.close()
                    break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-stage', type=int, default=0,
                       help='Which stage to start at.')
    parser.add_argument('-epochs', type=int, default=30,
                       help='Number of epochs')
    args = parser.parse_args()


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
            #Trainer(sample, 5, up, mask).startTraining(30, 1, "BBox_upsample/")
            Trainer(sample, 6, up, mask).startTraining(args.epochs, args.stage, "")
