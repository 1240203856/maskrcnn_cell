import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import references.detection.utils as utils
import references.detection.transforms as T
from references.detection.engine import train_one_epoch, evaluate
from dataloader import PennFudanDataset
from PIL import Image
import os
import numpy as np
import matplotlib as mpl
import numpy

###定义颜色
colorlist=['#000000','#FFFFFF']
def getcolor():
    colorArr=['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color=''
    for i in range(6):
        color+=colorArr[np.random.randint(1,15)]
    return '#'+color
def color():
    for i in range(10000):
        color=getcolor()
        colorlist.append(color)
    return colorlist

color=color()
    
# 自定义colormap
def colormap():
#    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', color, 256)

colormap=colormap()

#colormask=np.zeros((size,3))
color_pixel=[[0,0,0],[255,255,255]]
def randomcolor():
    for i in range(10000):
        color_RGB=[]
        R=np.random.randint(0,256)
        G=np.random.randint(0,256)
        B=np.random.randint(0,256)
        color_RGB.append(R)
        color_RGB.append(G)
        color_RGB.append(B)
        color_pixel.append(color_RGB)
    return color_pixel
randomcolor=randomcolor()

def pre_to_img(pre):
    #pre: numpy hxW
    print(pre.shape)
    row, col = pre.shape
    dst = numpy.zeros((row, col, 3), dtype=numpy.uint8)
    for i in range(10000):
        dst[pre == i] = randomcolor[i]
    dst = numpy.array(dst, dtype=numpy.uint8)
    image = Image.fromarray(dst)
    return image
