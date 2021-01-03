import numpy as np
import cv2
import matplotlib.pyplot as plt

from Classification_MNIST import Classification
from BoundingBox_Image import BoudingBox
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class SpeedTraceEstimator():
    varR = 1.
    varQ = 0.5
    test_ratio = 0.20

    def train(model=1, test_ratio=0.20):
        cl = Classification()
        cl.loadData("mnist_784", test_ratio=test_ratio)
        cl.trainModel(model=model)
        # sample = cl.getX_test()[36000]
        # print(np.shape(sample))
        # cl.predict(sample, 9)
        return cl

    def estimate(image, cl, bImageRead=1):
        H_IMG = 28
        OFFSET = 5

        bb = BoudingBox.run(image, bImageRead=bImageRead)

        arrDigit = [0] * len(bb)
        for i in range(0, len(bb)):
            image = bb[i]
            h = image.shape[0]
            w = image.shape[1]
            hResized = H_IMG-(OFFSET*2)  # offset similat to MNIST
            wResized = min(round(w/h * hResized), H_IMG)
            imageResized = cv2.resize(image, (wResized, hResized), interpolation=cv2.INTER_NEAREST)
            imageBlackBox = np.zeros((H_IMG, H_IMG))
            leftOffset = round((imageBlackBox.shape[1] - imageResized.shape[1])/2)
            y0, y1 = OFFSET, OFFSET+imageResized.shape[0]
            x0, x1 = leftOffset, leftOffset+imageResized.shape[1]
            imageBlackBox[y0: y1, x0: x1] = imageResized
            imageResizedReshaped = imageBlackBox.reshape(np.power(H_IMG, 2))
            # sample = cl.getX_test()[36000]
            sample = imageResizedReshaped
            digit = int(cl.predict(sample, 0)[0], base=10)
            arrDigit[i] = digit
        estimation = 0
        nDigit = len(arrDigit)
        for j in range(nDigit):  # left to right (ex.123: 1*10^2+2*10^1+3*10^0)
            nPower = nDigit - (j+1)
            estimation = estimation + arrDigit[j]*pow(10, nPower)
        print('Number estimation: ', estimation)
        return estimation

    def readVideo(self, videoPath, tStart=0, tEnd=1, downSample=1, model=1):
        FPS_VIDEO = 25
        cl = SpeedTraceEstimator.train(model=1, test_ratio=self.test_ratio)

        # KF instance and initialization
        f = KalmanFilter(dim_x=2, dim_z=1)
        varR = self.varR
        varQ = self.varQ
        f.x = np.array([[2.],    # position
                        [0.]])   # velocity
        f.F = np.array([[1., 1.],  # transition state matrix
                        [0., 1.]])
        f.H = np.array([[1., 0.]])  # state measure mat
        f.P = np.array([[1000.,    0.],  # just P init
                        [0., 1000.]])
        f.R = np.array([[varR]])  # measure noise cov mat
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=varQ)  # noise cov mat

        vidObj = cv2.VideoCapture(videoPath)
        nFrameStart = tStart*FPS_VIDEO
        nFrameEnd = tEnd*FPS_VIDEO
        arrSample = []
        arrEstimation = []
        arrEstimationKF = []

        success = 1
        i = 0
        while (success and (i <= nFrameEnd+1)):
            success, image = vidObj.read()
            if ((i % downSample == 0) and (i >= nFrameStart)):
                print(i)
                arrSample.append(i)
                image = [image][0]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # crop the image on speed dislpay
                # x0, y0, x1, y1 = 610, 570, 670, 610  # motogp
                # x0, y0, x1, y1 = 170, 420, 225, 445  # f1 rbr
                # x0, y0, x1, y1 = 145, 450, 200, 473 
                x0, y0, x1, y1 = 90, 342, 185, 380  # f1 abu
                imageCropped = image[y0:y1, x0:x1]
                # plt.imshow(imageCropped)
                # plt.show()
                # method call for each frame
                estimation = SpeedTraceEstimator.estimate(imageCropped, cl, bImageRead=0)
                arrEstimation.append(estimation)
                # apply KF
                z = estimation  # measure
                f.predict()
                f.update(z)
                x = f.x  # estimation from KF
                arrEstimationKF.append(x[0][0])

            i += 1
        # plot the result of the Estimtion
        plt.figure()
        plt.plot(arrSample, arrEstimation, label='measure', color='blue')
        plt.plot(arrSample, arrEstimationKF, label='estimation_KF', color='red', linewidth=3)
        plt.legend()
        plt.title('Speed Trace Estimation')
        plt.xlabel('Frame Sample')
        plt.ylabel('Speed kph')
        plt.show()


if __name__ == '__main__':
    image = 'speed.png'
    dte = SpeedTraceEstimator()
    # estimation = dte.estimate(image)
    # print('ESTIMATION: ', estimation)
    video = 'f1_Abu.mp4'
    dte.readVideo(video, tStart=0, tEnd=87, downSample=5, model=3)
