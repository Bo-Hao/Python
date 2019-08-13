import cv2
import numpy as np
import matplotlib.pyplot as plt

class KeypointsDetect():
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        img1 = cv2.imread(image1, 0)
        img2 = cv2.imread(image2, 0)
        self.imgQ = img1
        self.imgT = img2
        self.bool = 1 if image2 == None else 0
        self.keypoints = 'Not define yet'
        self.keypoints1 = 'Not define yet'
        self.keypoints2 = 'Not define yet'
 
    def ORB(self):
        orb = cv2.ORB_create()
        if self.bool == 1:
            self.keypoints = orb.detect(self.imgQ, None)
        elif self.bool == 0:
            self.keypoints1 = orb.detectAndCompute(self.imgQ, None)
            self.keypoints2 = orb.detectAndCompute(self.imgT, None)
            self.keypoints = [self.keypoints1, self.keypoints2]
        return self.keypoints
    
    def AKAZE(self):
    	akaze = cv2.AKAZE_create()
    	if self.bool == 1:
            self.keypoints = akaze.detect(self.imgQ, None)
    	elif self.bool == 0:
            self.keypoints1 = akaze.detectAndCompute(self.imgQ, None)
            self.keypoints2 = akaze.detectAndCompute(self.imgT, None)
            self.keypoints = [self.keypoints1, self.keypoints2]
    	return self.keypoints
    
    def BRISK(self):
        brisk = cv2.BRISK_create()
        if self.bool == 1:
            self.keypoints = brisk.detect(self.imgQ, None)
        elif self.bool == 0:
            self.keypoints1 = brisk.detectAndCompute(self.imgQ, None)
            self.keypoints2 = brisk.detectAndCompute(self.imgT, None)
            self.keypoints = [self.keypoints1, self.keypoints2]
        return self.keypoints
    
    def FAST(self):
        fast = cv2.FastFeatureDetector_create()
        if self.bool == 1:
            self.keypoints = fast.detect(self.imgQ, None)
        elif self.bool == 0:
            self.keypoints1 = fast.detect(self.imgQ, None)
            self.keypoints2 = fast.detect(self.imgT, None)
            br = cv2.BRISK_create();
            self.keypoints1 = br.compute(self.imgQ,  self.keypoints1)
            self.keypoint2s = br.compute(self.imgT,  self.keypoints2)
            
            self.keypoints = [self.keypoints1, self.keypoints2]
        return self.keypoints
    
    def KAZE(self):
    	kaze = cv2.KAZE_create()
    	if self.bool == 1:
            self.keypoints = kaze.detect(self.imgQ, None)
    	elif self.bool == 0:
            self.keypoints1 = kaze.detectAndCompute(self.imgQ, None)
            self.keypoints2 = kaze.detectAndCompute(self.imgT, None)
            self.keypoints = [self.keypoints1, self.keypoints2]
    	return self.keypoints
    
    def SUFT(self):
        surf = cv2.xfeatures2d.SURF_create()
        if self.bool == 1:
            self.keypoints = surf.detect(self.imgQ, None)
        elif self.bool == 0:
            self.keypoints1 = surf.detectAndCompute(self.imgQ, None)
            self.keypoints2 = surf.detectAndCompute(self.imgT, None)
            self.keypoints = [self.keypoints1, self.keypoints2]
        return self.keypoints
    
    def draw(self):
        if self.bool == 1:
            img = plt.imread(self.image1)
            plt.imshow(img)
            for i in self.keypoints:
                plt.scatter(i.pt[0], i.pt[1], s = 0.1)
        else:
            img1 = plt.imread(self.image1)
            img2 = plt.imread(self.image2)
            
            img3 = []
            max_row = max(len(img1), len(img2))
            for i in range(max_row):
                list1 = img1[i] if len(img1) > i else [[255, 255, 255] for i in range(len(img1[0]))]
                list2 = img2[i] if len(img2) > i else [[255, 255, 255] for i in range(len(img2[0]))]
                img3.append(list(list1) + list(list2))    
            plt.imshow(img3)
            
        plt.show()
        return img3

    def match(self):
        if self.bool == 1:
            print('Second image is needed!')
        else:
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
            clusters = np.array([self.keypoints[0][1]])
            bf.add(clusters)
            # Train: Does nothing for BruteForceMatcher though.
            bf.train()
            self.matches = bf.match(self.keypoints[1][1])
            self.matches = sorted(self.matches, key = lambda x:x.distance)
            
            pt1 = []
            pt2 = []
            for i in self.matches:
                pt1.append(self.keypoints[0][0][i.trainIdx].pt)
                pt2.append(self.keypoints[1][0][i.queryIdx].pt)
                
            self.pt1 = pt1
            self.pt2 = pt2
            
            
            return self.matches
        
    def drawmatch(self, N = 100):
        img1 = plt.imread(self.image1)
        img2 = plt.imread(self.image2)
        img3 = []
        max_row = max(len(img1), len(img2))
        for i in range(max_row):
            list1 = img1[i] if len(img1) > i else [[255, 255, 255] for i in range(len(img1[0]))]
            list2 = img2[i] if len(img2) > i else [[255, 255, 255] for i in range(len(img2[0]))]
            img3.append(list(list1) + list(list2))    
        plt.imshow(img3)
        for i in range(N):
            plt.scatter(np.array(K.pt1).T[0][i], np.array(K.pt1).T[1][i], c = 'yellow')
            plt.scatter(np.array(K.pt2).T[0][i] + len(img1[0]), np.array(K.pt2).T[1][i], c = 'red')
            x = [np.array(K.pt1).T[0][i], np.array(K.pt2).T[0][i] + len(img1[0])]
            y = [ np.array(K.pt1).T[1][i], np.array(K.pt2).T[1][i]]
            plt.plot(x, y)
        plt.savefig('/Users/pengbohao/Downloads/2019summer/img3.JPG')
        plt.show()
        
        
if __name__ == "__main__":
    image_address1 = '/Users/pengbohao/Downloads/2019summer/IMG_2881.JPG'
    image_address2 = '/Users/pengbohao/Downloads/2019summer/IMG_2882.JPG'
    #image_address1 = '/Users/pengbohao/Downloads/碩士班大頭貼.jpg'
    #image_address2 = '/Users/pengbohao/Downloads/高中大頭貼.jpg'
    K = KeypointsDetect(image_address1, image_address2)
    K.SUFT()

    K.match()
    K.drawmatch(N = 32)
    


