import numpy as np
import cv2


class BoudingBox():

    def run(imageName, bImageRead=1):

        # Load Image
        if bImageRead == 1:
            image = cv2.imread(imageName)
        else:
            image = imageName
        # swap black and white
        # im[im == 255] = 1
        # im[im == 0] = 255
        # im[im == 1] = 0

        # Convert image to gray scale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(imageGray, (5, 5), 0)
        ret, imageBinary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)     

        # Find contours
        contours, hierarchy = cv2.findContours(imageBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))

        arrBBox = []
        # Bouding Box on each features
        for i in range(0, len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            arrBBox.append([x, y, w, h])
        # sort Bounding Box left to right
        arrBBox.sort(key=lambda BBox: BBox[0])

        arrCroppedImage = []
        for i in range(0, len(arrBBox)):
            x, y, w, h = arrBBox[i]
            # print(np.shape(imageBinary))
            croppedImage = imageBinary[y:y+h, x:x+w]
            # print(np.shape(croppedImage))
            # cv2.imshow('Features', croppedImage)
            # arrCroppedImage = np.append(arrCroppedImage, croppedImage)
            arrCroppedImage.append(croppedImage)
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.imshow('Features', image)
            # cv2.imwrite(str(i)+'.png', im)
            # print('countor')
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print('BoudingBox done')

        # print(np.shape(arrCroppedImage))
        # print(arrCroppedImage)
        return arrCroppedImage

if __name__ == '__main__':
    BoudingBox.run('speed.png')
