import cv2
import dlib
import imutils


class FacePoseEstimator:
    def __init__(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def predict(self, imPath):
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
                x, y = shape.part(j).x*width/image.shape[1], shape.part(j).y*height/image.shape[0]
                faces[-1].append((int(x), int(y)))
        return faces


class BodyPoseEstimator:
    def __init__(self):
        # Specify the paths for the 2 files
        protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

        # Read the network into Memory
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    def predict(self, imPath):
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
        for i in range(len(18)):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            points.append((x, y))
        return points


if __name__ == "__main__":
    face = FacePoseEstimator()
    marks = face.predict("testFace.jpg")