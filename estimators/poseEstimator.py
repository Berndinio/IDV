########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################

import cv2
import dlib
import imutils
import numpy as np
import torch
from PIL import Image


class FacePoseEstimator:
    def __init__(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("pretrained_files/shape_predictor_68_face_landmarks.dat")

    def predict(self, imPath, show=False):
        faces = []
        image = cv2.imread(imPath)
        # reshape?
        height, width = image.shape[0], image.shape[1]
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        for (i, rect) in enumerate(rects):
            faces.append([])
            shape = self.predictor(gray, rect)
            for j in range(0, 68):
                x, y = shape.part(j).x * width / image.shape[1], shape.part(j).y * height / image.shape[0]
                faces[-1].append((int(x), int(y)))
        if show:
            image = cv2.imread(imPath)
            for i, (x, y) in enumerate(faces[0]):
                cv2.circle(image, (int(x), int(y)), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            cv2.imwrite("faceKeypoints.jpg", image)
        return faces

    def generateMask(self, keyPoints, imageShape, save=False):
        image = np.zeros(imageShape, dtype=np.uint8)
        sliced = keyPoints[slice(0, 17, 1)] + keyPoints[slice(26, 16, -1)]
        sliced = np.array([sliced])
        cv2.fillPoly(image, sliced, 1)
        # draw lines
        indiceList = list(range(0, 17)) + [0] + list(range(17, 27)) + [16]
        width = int((keyPoints[16][0] - keyPoints[0][0]) * 0.2)
        for j, kpIndex in enumerate(indiceList):
            if j == 0:
                continue
            idx1 = indiceList[j - 1]
            idx2 = indiceList[j]
            cv2.line(image, keyPoints[idx1], keyPoints[idx2], 1, width)

        if save:
            img = Image.fromarray(image * 255, 'L')
            img.save('faceMask.png')
        return torch.Tensor(image)


class BodyPoseEstimator:
    def __init__(self):
        # Specify the paths for the 2 files
        protoFile = "pretrained_files/COCO_pose_deploy_linevec.prototxt"
        weightsFile = "pretrained_files/COCO_pose_iter_440000.caffemodel"

        # Read the network into Memory
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    def predict(self, imPath, show=False):
        # Read image
        frame = cv2.imread(imPath)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        # Specify the input image dimensions
        inWidth = 368
        inHeight = 368
        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        # Set the prepared object as the input blob of the network
        self.net.setInput(inpBlob)
        output = self.net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []
        for i in range(18):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            points.append((int(x), int(y)))

        if show:
            for i, (x, y) in enumerate(points):
                cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 1,
                            lineType=cv2.LINE_AA)
            cv2.imwrite("bodyKeypoints.jpg", frame)
        return points

    def generateMask(self, keyPoints, imageShape, save=False):
        image = np.zeros(imageShape, dtype=np.uint8)
        # draw polygons
        polygons = [[], []]
        for i in [16, 15, 17, 1]:
            polygons[0] += [keyPoints[i]]
        for i in [2, 5, 11, 8]:
            polygons[1] += [keyPoints[i]]
        print(polygons)
        polygons = np.array(polygons)
        cv2.fillPoly(image, polygons, 1)
        # draw lines
        indiceList = [[10, 9, 8, 11, 12, 13], [11, 5, 6, 7], [8, 2, 3, 4], [2, 5], [1, 16, 15, 17, 1]]
        width = int((keyPoints[11][0] - keyPoints[8][0]) * 0.75)

        for i, sublist in enumerate(indiceList):
            for j, kpIndex in enumerate(sublist):
                if j == 0:
                    continue
                idx1 = indiceList[i][j - 1]
                idx2 = indiceList[i][j]
                cv2.line(image, keyPoints[idx1], keyPoints[idx2], 1, width)
        # save it
        if save:
            img = Image.fromarray(image * 255, 'L')
            img.save('bodyMask.png')
        return torch.Tensor(image)


if __name__ == "__main__":
    face = FacePoseEstimator()
    marks = face.predict("testFace.jpg", True)
    mask = face.generateMask(marks[0], (3000, 2391), True)
    print("Facial landmarks:", marks)

    body = BodyPoseEstimator()
    marks = body.predict("testPerson.jpg", True)
    mask = body.generateMask(marks, (2048, 1360), True)
    print("Body landmarks:", marks)
