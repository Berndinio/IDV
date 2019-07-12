########################################################################################################################
# Jan Sieber, 3219317
# Python 3.5
# Deep Vision, University of Heidelberg, Prof. Dr. Björn Ommer
########################################################################################################################

import copy

import cv2
import dlib
import numpy as np
import torch
from PIL import Image

class FacePoseEstimator:
    def __init__(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("estimators/pretrained_files/shape_predictor_68_face_landmarks.dat")

    def predict(self, image, show=False):
        faces = []
        # reshape?
        height, width = image.shape[0], image.shape[1]
        tmpImage = copy.deepcopy(image)
        #image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) > 0:
            for (i, rect) in enumerate(rects):
                faces.append([])
                shape = self.predictor(gray, rect)
                for j in range(0, 68):
                    x, y = shape.part(j).x * width / image.shape[1], shape.part(j).y * height / image.shape[0]
                    faces[-1].append((int(x), int(y)))
            if show:
                image = tmpImage
                for i, (x, y) in enumerate(faces[0]):
                    cv2.circle(image, (int(x), int(y)), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                cv2.imwrite("estimators/images/faceKeypoints.jpg", image)
        return faces, rects

    def generateMask(self, keyPoints, cv_image, save=False):
        image_new = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.float)
        sliced = keyPoints[slice(0, 17, 1)] + keyPoints[slice(26, 16, -1)]
        sliced = np.array([sliced])
        cv2.fillPoly(image_new, sliced, 1)
        # draw lines
        indiceList = list(range(0, 17)) + [0] + list(range(17, 27)) + [16]
        width = int((keyPoints[16][0] - keyPoints[0][0]) * 0.2)
        for j, kpIndex in enumerate(indiceList):
            if j == 0:
                continue
            idx1 = indiceList[j - 1]
            idx2 = indiceList[j]
            cv2.line(image_new, keyPoints[idx1], keyPoints[idx2], 1, width)

        if save:
            img = Image.fromarray(image_new * 255, 'L')
            img.save('estimators/images/faceMask.png')
        image_new = np.repeat(image_new[None], cv_image.shape[2], axis=0)
        return torch.Tensor(image_new)

    def generatePoseEmbedding(self, keypoints, image, save=False):
        heatmaps = np.zeros((len(keypoints), image.shape[0], image.shape[1]), dtype=np.float)
        for i, p in enumerate(keypoints):
            cv2.circle(heatmaps[i], p, 5, 1, thickness=-1, lineType=cv2.FILLED)
        if save:
            for i in range(len(keypoints)):
                img = Image.fromarray(heatmaps[i] * 255, 'L')
                img.save("estimators/images/HM" + str(i) + ".png")
        return torch.Tensor(heatmaps)

    def generateBboxMask(self, bbox, cv_image, save=False):
        image_new = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.float)
        center = (int(bbox.left() + bbox.width() / 2.0),
                  int(bbox.top() + bbox.height() / 2.0 - 0.1 * bbox.height()))
        cv2.ellipse(image_new, center, (int(bbox.width() * 1.5), int(bbox.height() * 1.2)), 0, 0, 360, 1, thickness=-1,
                    lineType=cv2.FILLED)
        if save:
            img = Image.fromarray(image_new * 255, 'L')
            img.save('estimators/images/faceMaskBox.png')
        image_new = np.repeat(image_new[None], cv_image.shape[2], axis=0)
        return torch.Tensor(image_new)

    def predictPose(self, im, keypoints):
        size = im.shape

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            keypoints[30],  # Nose tip
            keypoints[8],  # Chin
            keypoints[36],  # Left eye left corner
            keypoints[45],  # Right eye right corne
            keypoints[48],  # Left Mouth corner
            keypoints[54]  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # Camera internals

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        ret = (rotation_vector, translation_vector)
        return ret


class BodyPoseEstimator:
    def __init__(self):
        # Specify the paths for the 2 files
        protoFile = "estimators/pretrained_files/COCO_pose_deploy_linevec.prototxt"
        weightsFile = "estimators/pretrained_files/COCO_pose_iter_440000.caffemodel"

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
            cv2.imwrite("estimators/images/bodyKeypoints.jpg", frame)
        return points

    def generateMask(self, keyPoints, imageShape, save=False):
        image = np.zeros(imageShape, dtype=np.float)
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
            img.save('estimators/images/bodyMask.png')
        return torch.Tensor(image)


if __name__ == "__main__":
    import os

    # Body pose
    body = BodyPoseEstimator()
    marks = body.predict("estimators/testPerson.jpg", True)
    mask = body.generateMask(marks, (128, 64), True)
    print("Body landmarks:", marks)

    ffiles = []
    for root, dirs, files in os.walk("dataset/FEI"):
        ffiles = files
        if ".keep" in ffiles:
            ffiles.remove(".keep")
        continue
    face = FacePoseEstimator()
    for file in ffiles:
        path = root + "/" + file
        print("Processing " + path)
        image = cv2.imread(path)
        faces, rects = face.predict(image, True)
        if len(faces) > 0:
            # for f in faces:
            #    pose = face.predictPose(image, f)
            #    print(pose)
            mask = face.generateMask(faces[0], image, True)
            mask = face.generateBboxMask(rects[0], image, True)
            embedding = face.generatePoseEmbedding(faces[0], image, True)
        else:
            print("ERROR with " + path)
