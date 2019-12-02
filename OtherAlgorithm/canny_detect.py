import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
from PIL import Image


def detect(image):
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    canny = cv2.Canny(blurred, 30, 90)

    plt.imshow(canny)
    plt.show()






if __name__ == "__main__":
    image = '/Users/pengbohao/Downloads/IMG_1304.JPG'
    detect(image)