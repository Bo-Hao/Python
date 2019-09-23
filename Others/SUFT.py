import cv2
import numpy
 
 
def main():
 
 
	img = cv2.imread("/Users/pengbohao/Downloads/2019summer/IMG_2881.JPG")
	cv2.imshow('Input Image', img)
	cv2.waitKey(0)
	
	# 检测
	orb = cv2.ORB_create()
	keypoints = orb.detect(img, None)
	
	# 显示
	# 必须要先初始化img2
	img2 = img.copy()
	img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0,255,0))
	cv2.imshow('Detected ORB keypoints', img2)
	cv2.waitKey(0)


main()
cv2.destroyAllWindows()