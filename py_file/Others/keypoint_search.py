from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def keypoint_search(image1, image2, N = 10):
	img1 = cv2.imread(str(image1),0)        # queryImage
	img2 = cv2.imread(str(image2),0) 		# trainImage

	col1, row1 = img1.shape[1], img1.shape[0]
	col2, row2 = img2.shape[1], img2.shape[0]

	#M = cv2.getRotationMatrix2D((col2/2,row2/2),90,1)
	#img2 = cv2.warpAffine(img2,M,(col2,row2))
	
    # Initiate SIFT detector
	orb = cv2.ORB_create()
	

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)


	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:N], None,  flags=2)



	# Initialize lists
	list_kp1 = []
	list_kp2 = []

	# For each match...
	for mat in matches[:N]:

	    # Get the matching keypoints for each of the images
	    img1_idx = mat.queryIdx
	    img2_idx = mat.trainIdx

	    # x - columns
	    # y - rows
	    # Get the coordinates
	    (x1,y1) = kp1[img1_idx].pt
	    (x2,y2) = kp2[img2_idx].pt


	    # Append to each list
	    list_kp1.append((x1, y1))
	    list_kp2.append((x2, y2))

    
	list_kp1, list_kp2 = np.array(list_kp1), np.array(list_kp2)

	plt.imshow(img3)#, plt.show()


	return list_kp1, list_kp2

def image_movement(pt1, pt2, method = cv2.FM_8POINT):
	F, mask = cv2.findFundamentalMat(pt1, pt2, method)
	
	u, d, v = np.linalg.svd(F)
	Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	S = np.dot(np.dot(u, Z), np.transpose(u))
	R = np.dot(np.dot(u, W), v)

	displacement = [S[1][2], S[2][0], S[0][1]]
	return displacement


sca = np.array(Image.open('/Users/pengbohao/Downloads/2019summer/IMG_2881.JPG')).shape[1]
pt1, pt2 = keypoint_search('/Users/pengbohao/Downloads/2019summer/IMG_2881.JPG', '/Users/pengbohao/Downloads/2019summer/IMG_2882.JPG', N = 200)
plt.scatter(pt1.T[0], pt1.T[1], c = 'red', s = 0.1)
plt.scatter(pt2.T[0]+sca, pt2.T[1], c = 'yellow', s = 0.1)
plt.savefig('/Users/pengbohao/Downloads/2019summer/save.JPG', quality = 95, dpi = 1000)
plt.show()



