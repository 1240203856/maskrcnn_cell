#coding:utf-8
import cv2
import os
from PIL import Image
import numpy as np
import math

#overlap_half=True滑动窗口切图，每次有一半区域重叠，这时候x方向的步长就是窗口宽度的一半，y方向的步长是窗口高度的一半，stridex和stridey参数将不再起作用
def slide_crop(img,image_name,kernelw,kernelh,mode,savePath=None,overlap_half=True,stridex=0,stridey=0):
    if os.path.exists(savePath) is False:
        os.makedirs(savePath)
		
    height= img.shape[0]
    width = img.shape[1]
    if overlap_half:
#        stridex = 200
#        stridex=186
        stridex=200
#        stridey = 200
#        stridey=186
        stridey=200
    img_list = []
    stepx = math.ceil(width / stridex)
    stepy = math.ceil(height / stridey)
    print(stepx,stepy)
    for r in range(stepy-1):
        startx = 0
        starty = r * stridey
        for c in range(stepx-1):
            startx = c*stridex
            img_list.append(img[starty:starty+kernelh,startx:startx+kernelw])
    i=0
    for image in img_list:
        i+=1
        if mode=='image':
            image=Image.fromarray(image)
            image.save(savePath+str(i)+'-'+str(image_name))
        if mode=='label':
            np.save(savePath+str(i)+'-'+str(image_name).split('.')[0]+'.npy',image)
		
def get_image(root):
    root=root
    image_dir=os.path.join(root,'Images')
    label_dir=os.path.join(root,'Labels')
    image_list=[]
    label_list=[]
    for image in os.listdir(image_dir):
        image_list.append(image)
    for label in os.listdir(label_dir):
        label_list.append(label)
    for i in range(len(image_list)):
        image_name=os.path.join(image_dir,image_list[i])
        label_name=os.path.join(label_dir,image_list[i].split('.')[0]+'.npy')
#        label_name=os.path.join(label_dir,'labels-'+image_list[i].split('pred-')[1])
        image=Image.open(image_name)
#        label=Image.open(label_name)
        label=np.load(label_name)
        image=np.array(image)
        label=np.array(label)
        print(image.shape,label.shape)
        slide_crop(img=image,image_name=image_list[i],kernelw=200,kernelh=200,mode='image',savePath='./patch_data_test/diff_tissue/Images/',overlap_half=True,stridex=0,stridey=0)
        slide_crop(img=label,image_name=image_list[i],kernelw=200,kernelh=200,mode='label',savePath='./patch_data_test/diff_tissue/Labels/',overlap_half=True,stridex=0,stridey=0)

if __name__ == "__main__":
    ROOT='./data-ins/test/diff_tissue/'
#    ROOT='../valid/same/pred/'
    get_image(ROOT)


