'''
    Websites that helped me build the data set and can help in the future to improve
    https://github.com/utkuozbulak/pytorch-custom-dataset-examples
    https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://pytorch.org/docs/stable/data.html
'''

'''
    Every time that I want to change this code I need to first delete the __pycash__
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
        
        
        """
        Args:
            csv_path (string): path to csv file
        """


        
        #------------------Crop the Images-Video---------------
        # I need to change the __init__ in order to take this information from the 
        # data loader in the 1_dataLoaderFromDataset.ipynb file 
        
        self.startH = startH
        self.startW = startW
        
        self.heightVid = heightVid
        self.widthVid = widthVid

        self.frameStart = frameStart
        self.frames = frames
        self.csv_path = csv_path

        #------------------End Crop the Images-Video---------------
        
        
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
        
        container = av.open(single_video_name) #open the whole video
        # This is propably a bad idea because it is really slow, I need to find a way to create
        # something faster

        arr = np.empty((self.frames,self.heightVid,self.widthVid,3),np.dtype('uint8'))
        
        i=0 # Number of Frame 
        for frame in container.decode(video=0):

            if (i>=self.frameStart):


                img=frame.to_image() # read one frame

                arr2 = np.asarray(img) # Change it to numpy

                # arr2= np.einsum('kli->ikl', arr2)
                #
                # print(arr2.shape)
                # -----------Resize the Images------------------
                # arr2 = cv2.resize(arr2, dsize=(self.widthVid, self.heightVid), interpolation=cv2.INTER_AREA)
                # ----------------------------------------------

                #-----------Crop the Images--------------------
                arr2 = arr2[self.startH:self.startH+self.heightVid , self.startW:self.startW+self.widthVid , :] # Crop the image
                # print(arr.shape)
                #----------------------------------------------

                point = i-self.frameStart

                arr[point,:,:,:] = arr2

            i = i + 1

            if (i>=self.frameStart+self.frames):
                break

        arr= np.einsum('akli->iakl', arr)
         
        videoLabel = self.label_arr[index] # take the label of the videos 
        
        return (arr, videoLabel) # I always need to give back only two thinks the image-video
                                 # And the label the rest will be taken care off by the system
        
       
    def __len__(self): # This need to exist from the documentation
        return self.data_len
        
        
        