import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import tkinter as tk 
from PIL import Image

def onclick(event):
	if event.xdata != None and event.ydata != None :
		print(event.xdata, event.ydata)

def clickfun(img):
	im=Image.open(img)
	ax = plt.gca()
	fig = plt.gcf()
	implot = ax.imshow(im)
	cid = fig.canvas.mpl_connect('key_press_event', onclick)
	plt.show()


if __name__=='__main__':
	pass