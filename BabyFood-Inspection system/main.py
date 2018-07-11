#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:35:47 2018

@author: bhargav
"""

import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt



def closeWindow():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def load_sample():
    #load the sample images
    #convert to hsv space
    #compute mean saturation
    #return the mean saturation of input images
    identifier='BabyFood-Sample'    
    set_sample=glob.glob(identifier+'*')
    out=[]
    for i in range(len(set_sample)):
        sample=cv2.imread(identifier+str(i)+'.JPG')
        hsv=cv2.cvtColor(sample,cv2.COLOR_BGR2HSV)
        hsv_mean=np.mean(hsv[:,:,1].astype(np.float32))
        out.append(hsv_mean)
    return(np.asarray(out,dtype=np.float32))

    
    
def evaluate_test(out_sample):
    #compute the mean saturation of the current image
    #compare with the sample images
    #return the most likely output
    identifier='BabyFood-Test'
    set_test=glob.glob(identifier+'*')
    out=[]
    for i in range(len(set_test)):
        sample=cv2.imread(identifier+str(i+1)+'.JPG')
        hsv=cv2.cvtColor(sample,cv2.COLOR_BGR2HSV)
        hsv_mean=np.mean(hsv[:,:,1].astype(np.float32))
        out.append(np.argmin((out_sample-hsv_mean)**2))
    return(np.asarray(out,dtype=np.float32))
    
def plot(index,final_result):
    #the first argument takes the integer value n
    #computes the n+1th image plots
    identifier='BabyFood-Test'
    sample=cv2.imread(identifier+str(index+1)+'.JPG')
    f,axarr=plt.subplots(2,sharex=True)
    axarr[0].imshow(sample)
    axarr[0].set_title("Image")
    axarr[1].text(sample.shape[1]/2,0.5,final_result[index],size=50,rotation=0,ha='center'
         ,va='center',color='r')
    
    

    
if __name__ == '__main__':
    
    output=load_sample()
    final_result=evaluate_test(output)
    #plot the results
    
    plot(2,final_result)
    
    
    
    



