import cv2




class PoseEstimator:
    def __init__(self):
        # Specify the paths for the 2 files
        protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

        # Read the network into Memory
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    def predict(self, imPath):

        # Read image
        frame = cv2.imread("single.jpg")
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