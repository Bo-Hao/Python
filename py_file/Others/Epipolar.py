import numpy as np 
import matplotlib as plt 
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
Image.MAX_IMAGE_PIXELS = 500000000



class Clickit():
    def __init__(self, image_address):
        self.image_address = image_address
        self.coordinate = []
        self.image_information = []
        self.draw_set = []

    def clickfun(self):
        def onclick(event):
            if event.xdata != None and event.ydata != None :
                if event.key == " ":
                    #print(event.xdata, event.ydata)
                    self.coordinate.append([event.xdata,event.ydata])
                    draw_event = plt.scatter(event.xdata, event.ydata, color = "red", s = 7)
                    self.draw_set.append(draw_event)
                    plt.draw()
                if event.key == "d" and len(self.draw_set) > 0:
                    self.draw_set[-1].remove()
                    del self.draw_set[-1]
                    del self.coordinate[-1]
                    plt.draw()
                if event.key == "n":
                    plt.close()
                    
        image_object = Image.open(self.image_address)
        image_object = image_object.convert("RGBA")
        plt.gca().imshow(image_object)
        plt.gcf().canvas.mpl_connect("key_press_event", onclick)
        plt.ion()
        
        #ax.imshow(image_object)
        #fig.canvas.mpl_connect("key_press_event", onclick)
        plt.show()
        return self.coordinate, np.array(image_object).shape
    
class Epipolar:
    def __init__(self, points_1, points_2):
        self.pt1 = points_1
        self.pt2 = points_2

    def epipolar(self):
        A = [[self.pt1[k][i]*self.pt2[k][j]for i in np.arange(3)for j in np.arange(3)]for k in np.arange(len(self.pt1))]
        A = np.array(A)
        U, D, Vt = np.linalg.svd(A)
        u, d, vt = np.linalg.svd(np.reshape(Vt[-1], (3, 3)))
        d = np.diag([d[0], d[1], 0])
        F = np.dot(np.dot(u, d), v)
        return F
    
    
        
        
        
        
        
        
        
    
    
    
    