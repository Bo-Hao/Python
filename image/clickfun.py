import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import tkinter as tk 
from PIL import Image
import numpy as np  
import pylab as plb

def onclick(event):
	if event.xdata != None and event.ydata != None :
		if event.key == " ":
			print(event.xdata, event.ydata)
			pt.append([event.xdata,event.ydata])
			plb.scatter(event.xdata, event.ydata, color = "red", s = 7)
			plb.show()
		elif event.key == "d":
			plt.close()
		else:
			pass

def clickfun(img):
	global pt, ptG, ptT, mpt
	pt = []
	im = Image.open(img)
	ax = plt.gca()
	fig = plt.gcf()
	plt.ion()
	implot = ax.imshow(im)
	cid = fig.canvas.mpl_connect('key_press_event', onclick)
	plt.show()


	pixel = np.array(im).shape
	return pt, pixel

#clickfun(imgage)
