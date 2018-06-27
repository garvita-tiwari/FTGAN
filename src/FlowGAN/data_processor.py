
from __future__ import print_function

import chainer
import random
import numpy as np
import scipy.misc
import os
from scipy.io import loadmat
import ipdb
class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, root_dir, crop_size=(64, 64), video_len=32):
        self.dataset = dataset
        self.root_dir = root_dir
        self.crop_x = crop_size[0]
        self.crop_y = crop_size[1]
        self.video_len = video_len
        self.stride = self.video_len//2

    def __len__(self):
        return len(self.dataset)

    def preprocess_img(self, flow, flipcrop=True):
        crop_x = self.crop_x
        crop_y = self.crop_y
        ipdb.set_trace()
         #2,batchsize,img_dim
        _, _, h, w = flow.shape
        #Why do we need flipcrop
        if flipcrop:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_y - 1)
            left = random.randint(0, w - crop_x - 1)
            if random.randint(0, 1):
                flow = flow[:, :, :, ::-1]
                flow[0,:,:,:] = 255 - flow[0,:,:,:]
            bottom = top + crop_y
            right = left + crop_x
            flow = flow[:,:, top:bottom, left:right]
        else:
            dst_flow = np.zeros((2,flow.shape[1],self.crop_y, self.crop_x), np.uint8)
            for i in range(flow.shape[1]):
                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
            flow = dst_flow
        flow = flow.astype(np.float32) * (2 / 255.) - 1.
        return flow

    def get_example(self, i, train=True):
        while 1:
            if os.path.exists(self.root_dir + self.dataset[i] + '.npy'):
                video_flow = np.load(self.root_dir + self.dataset[i] + '.npy')
                N = video_flow.shape[0]
                try:
                    if N == 32:
                        j = 0
                    else:
                        j = np.random.randint(0, N - self.video_len)
                except:
                    i = np.random.randint(len(self.dataset))
                    continue
                break
            else:
                print('no data',self.dataset[i])
                i = np.random.randint(len(self.dataset))
        ipdb.set_trace()
        Flows = video_flow[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X)
        flow = self.preprocess_img(Flows, flipcrop=True)

        return flow

class PreprocessedDataset_pose(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, root_dir, crop_size=(64, 64), video_len=32):
        self.dataset = dataset
        self.root_dir = root_dir
        self.crop_x = crop_size[0]
        self.crop_y = crop_size[1]
        self.video_len = video_len
        self.stride = self.video_len//2

    def __len__(self):
        return len(self.dataset)

    def preprocess_img(self, pose, flipcrop=False):
        # Why crop_x, crop_y  is 64,64
        crop_x = self.crop_x
        crop_y = self.crop_y
        flipcrop = False
        ipdb.set_trace()
        _, num_frames, num_keypoints = pose.shape  #_,_,h,w
        # why do we need flipcrop
        if flipcrop:
            print('nothing')
            # Randomly crop a region and flip the image
            #top = random.randint(0, h - crop_y - 1)
            #left = random.randint(0, w - crop_x - 1)
            #if random.randint(0, 1):
            #    flow = flow[:, :, :, ::-1]
            #    flow[0,:,:,:] = 255 - flow[0,:,:,:]
            #bottom = top + crop_y
            #right = left + crop_x
            #flow = pose[:,:, top:bottom, left:right]
        else:
            dst_pose = np.zeros((2,pose.shape[1],self.crop_y, self.crop_x), np.uint8) # for video.len
            for i in range(pose.shape[1]):
                dst_pose[0, i] = scipy.misc.imresize(pose[0, i, :], [self.crop_y, self.crop_x], 'bicubic')
                dst_pose[1, i] = scipy.misc.imresize(pose[1, i, :], [self.crop_y, self.crop_x], 'bicubic')
                #dst_pose[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                #dst_pose[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
            pose = dst_pose
        pose = pose.astype(np.float32)
        return pose

    def get_example(self, i, train=True):
        while 1:
            print(self.root_dir + self.dataset[i] + '.mat')
            if os.path.exists(self.root_dir + self.dataset[i] + '.mat'):
                pose_data = loadmat(self.root_dir + self.dataset[i] + '.mat')
                x_data = pose_data['x']
                y_data = pose_data['y']

                N = pose_data['dimensions'][0][2]
                   #if N= 32, then there are 32 frames, so take all the frame, else randmoly choose 32 frames
                try:
                    if N == 32:
                        j = 0
                    else:
                        j = np.random.randint(0, N - self.video_len)
                except:
                    i = np.random.randint(len(self.dataset))
                    continue
                break
            else:
                print('no data',self.dataset[i])
                i = np.random.randint(len(self.dataset))

        #Flows = pose_data[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X) , this line gives unhashable error, because pose_data has x,y,dimenstions and many dicts
        # above line changes (32, 76, 76, 2) to (2, 32, 76, 76)
        Poses = np.array([x_data[j:j+self.video_len],y_data[j:j+self.video_len]])
        #Poses =  np.concatenate((x_data[j:j+self.video_len],y_data[j:j+self.video_len] ), axis=0)
        pose = self.preprocess_img(Poses, flipcrop=True)

        return pose
