import numpy as np 
import matplotlib
matplotlib.use("TkAgg")
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt 
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500000000



class Clickit():
    def __init__(self, image_address):
        self.image_address = image_address
        self.coordinate = []
        self.image_information = []
        self.draw_set = []
        image_object = Image.open(self.image_address)
        image_object = np.array(image_object.convert("RGBA"))
        self.shape = image_object.shape
        
                
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
        ax = plt.gca()
        fig = plt.gcf()
        ax.imshow(image_object)
        cid = fig.canvas.mpl_connect('key_press_event', onclick)
        plt.show()
        
        
        
        #plt.ion()
        #plt.gca().imshow(image_object)
        #plt.gcf().canvas.mpl_connect("key_press_event", onclick)
        
        
    
        #ax.imshow(image_object)
        #fig.canvas.mpl_connect("key_press_event", onclick)

        #self.shape = np.array(image_object).shape
        return self.coordinate



C1 = Clickit('/Users/pengbohao/Downloads/2019summer/IMG_2881.JPG')
c1 = C1.clickfun()

C2 = Clickit('/Users/pengbohao/Downloads/2019summer/IMG_2882.JPG')
c2 = C2.clickfun()


import pickle
with open('/Users/pengbohao/Downloads/2019summer/save.pickle', 'wb') as f:
    pickle.dump([c1, c2], f)






# /Users/pengbohao/Downloads/2019summer/IMG_2881.JPG