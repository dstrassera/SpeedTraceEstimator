import numpy as np
import cv2
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from sklearn.metrics import mean_absolute_error

from Classification_MNIST import Classification
from BoundingBox_Image import BoudingBox


class SpeedTraceEstimator():
    FPS_VIDEO = 25
    varR = 3
    varQ = 10
    test_ratio = 0.20
    # labelled data
    labelled_downSample = 50
    labelled_data = [266, 286, 168, 216, 254, 269, 275, 286, 246, 121, 114, 105, 72, 156, 235, 268, 296, 324, 145, 71, 116, 192, 250, 286, 306, 318, 246, 103, 126, 158, 122, 165, 233, 263, 274, 110, 129, 127, 145, 174, 236, 217, 246, 140]

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
            # plt.imshow(imageBlackBox, cmap='gray')
            # plt.show()
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
        nFrameStart = tStart*self.FPS_VIDEO
        nFrameEnd = tEnd*self.FPS_VIDEO
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

        # labelled data
        step = self.labelled_downSample
        stop = len(self.labelled_data)
        arrSampleLabelled = np.arange(0, stop*step, step)

        # subsample estimation as the labelled data
        if ((self.labelled_downSample % downSample) == 0):
            downSample_ratio = round(self.labelled_downSample/downSample)
            # numpy slicing [start:stop:step]
            arrEstimationKF_sub = arrEstimationKF[0:len(arrEstimationKF):downSample_ratio]
            MAE = round(mean_absolute_error(self.labelled_data, arrEstimationKF_sub), 1)
            print('Mean Absolute Error: ', MAE)
        else:
            print('CANT CALCULATE ERROR IF DOWNSAMPLE OF LABELLED IS A SUB')

        # plot the result of the Estimtion
        plt.figure()
        plt.plot(arrSample, arrEstimation, label='measure', color='blue')
        plt.plot(arrSample, arrEstimationKF, label='estimation_KF', color='red', linewidth=3)
        plt.plot(arrSampleLabelled, self.labelled_data, 'go', label='labelled')
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
