import numpy as np
import cv2
from matplotlib import pyplot as plt

# img1 = cv2.imread(queryImage)
# img2 = cv2.imread(trainImage)
def auto_keypoint_searching(img1, img2):
    # 仿射旋轉 
    cols = img1.shape[1]
    rows = img1.shape[0]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    img1 = cv2.warpAffine(img1,M,(cols,rows))

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40], None,  flags=2)

    # Initialize lists
    list_kp1 = []
    list_kp2 = []

    # For each match...
    for mat in matches[:40]:

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
    F = cv2.findFundamentalMat(list_kp1, list_kp2, cv2.FM_RANSAC)[0]
    u, d, v = np.linalg.svd(F)
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    S = np.dot(np.dot(u, Z), np.transpose(u))
    R = np.dot(np.dot(u, W),v )
    displace = [[S[1][2], S[2][0], S[0][1]]]


    print(cv2.RQDecomp3x3(R))
    print(displace)


    plt.imshow(img3), plt.show()
