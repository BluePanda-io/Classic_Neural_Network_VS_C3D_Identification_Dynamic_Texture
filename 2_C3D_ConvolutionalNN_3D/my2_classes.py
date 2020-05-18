'''
    Every time that I want to change this code I need to first delete the __pycash__ or else my2_classes.pyc
    because this code is basically used in the 1_dataLoaderFromDataset.ipynb, so need to be sincronized
    with the new one 
'''

import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np

import av

import cv2


class customDatasetVideos(Dataset):
    
    def __init__(self,csv_path, frameStart, frames, startH,startW,heightVid,widthVid):

        #------------------Initialize Crop the Images-Video---------------
        # I need to change the __init__ in order to take this information from the 
        # data loader in the 1_dataLoaderFromDataset.ipynb file 
        
        self.startH = startH
        self.startW = startW
        
        self.heightVid = heightVid
        self.widthVid = widthVid

        self.frameStart = frameStart
        self.frames = frames
        #------------------End Crop the Images-Video---------------
        

        self.csv_path = csv_path

        # Read the csv files with all the apths and all the corresponding labels
        self.data_info = pd.read_csv(self.csv_path, header=None)
        
        # First column contains the image paths
        self.video_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third  column is the frames
        self.frames_arr = np.asarray(self.data_info.iloc[:, 2])

        self.data_len = len(self.data_info.index) # I don't know why I do that it was in the documentation

        
    def __getitem__(self,index):
        
        single_video_name = self.video_arr[index]# get the path 
        
        cap = cv2.VideoCapture(single_video_name) #open the whole video
        # This is propably a bad idea because it is really slow, I need to find a way to create
        # something faster

        arr = np.empty((self.frames,self.heightVid,self.widthVid,3),np.dtype('uint8'))


        i=0 # Number of Frame
        newVideo=0
        flagCorruptedVideo=0
        while True:
            success, img = cap.read()  # propablly problem with the successf
            #print(i,success,single_video_name)

            if success: # success==True # This means that everything works so we will make the flag 0
                flagCorruptedVideo = 0
            else: # This means that we have a problem so we will start counting
                flagCorruptedVideo += 1

            if flagCorruptedVideo==5: # If we cound 5 problems in a row this means that the video is propably corrupted so we will go to the next one
                newVideo+=1
                single_video_name = self.video_arr[index+newVideo]  # There is propably a more scientific way to do that by for now this just works

                cap = cv2.VideoCapture(single_video_name)

            if i>=self.frameStart:

                if success: # If Seccess==True

                    arr2 = np.asarray(img)

                    # -----------Resize the Images------------------
                    arr2 = cv2.resize(arr2, dsize=(self.widthVid, self.heightVid), interpolation=cv2.INTER_AREA)
                    # ----------------------------------------------
                    #-----------Crop the Images--------------------
                    # arr2 = arr2[self.startH:self.startH+self.heightVid , self.startW:self.startW+self.widthVid , :] # Crop the image
                    #----------------------------------------------

                    point = i-self.frameStart
                    arr[point,:,:,:] = arr2
                else: # If the frame is broken we will go back a step and dont recognize it as seperate entity # we will search again for the next frame
                    i=i-1
            i = i + 1

            if (i>=self.frameStart+self.frames): # SOS there is a better way to do that instead of break
                break

        cap.release()

        arr = np.einsum('akli->iakl', arr) # Change everytihng together, it is really important to have the channels first and then frames Hieght Width
         
        videoLabel = self.label_arr[index] # take the label of the videos 
        
        return (arr, videoLabel) # I always need to give back only two thinks the image-video
                                 # And the label the rest will be taken care off by the system
        
       
    def __len__(self): # This need to exist from the documentation
        return self.data_len
        
        
        