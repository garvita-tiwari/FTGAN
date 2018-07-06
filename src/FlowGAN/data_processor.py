
from __future__ import print_function

import chainer
import random
import numpy as np
import scipy.misc
import os
from scipy.io import loadmat
import ipdb
import cv2


def ConvertFlow2Img(Flows, lower_bound, higher_bound):
    DstFlows = (Flows - lower_bound) / float(higher_bound - lower_bound) * 255.
    DstFlows = np.round(DstFlows).astype(np.uint8)

    low_index = np.where(Flows < lower_bound)
    DstFlows[low_index] = 0
    high_index = np.where(Flows > higher_bound)
    DstFlows[high_index] = 255

    return DstFlows
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
        flipcrop=False
        crop_x = self.crop_x
        crop_y = self.crop_y
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
            dst_flow = np.zeros((3,flow.shape[1],self.crop_y, self.crop_x), np.uint8)
            for i in range(flow.shape[1]):
                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[2, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x], 'bicubic')
            flow = dst_flow
        else:
            dst_flow = np.zeros((3,flow.shape[1],self.crop_y, self.crop_x), np.uint8)
            check2 = np.zeros((self.crop_y, self.crop_x), np.uint8)
            for i in range(flow.shape[1]):

                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[2, i] = scipy.misc.imresize(check2, [self.crop_y, self.crop_x], 'bicubic')
            flow = dst_flow
        flow = flow.astype(np.float32) * (2 / 255.) - 1.
        return flow

    def get_example(self, i, train=True):
        while 1:
            if os.path.exists(self.root_dir + self.dataset[i] + '.npy'):
                video_flow = np.load(self.root_dir + self.dataset[i] + '.npy')
                """
                video_flow = np.load('/BS/vedika/nobackup/flownet2-pytorch/result_UFC101/inference/run.epoch-0-flow-field.npy')
                N = video_flow.shape[0]
                for i in range(1,N):
                    bound = 20
                    x_flow = video_flow[i,:,:,0]
                    x_flow = ConvertFlow2Img(x_flow, -1 * bound, bound)

                    y_flow = video_flow[i,:,:,1]
                    y_flow = ConvertFlow2Img(y_flow, -1 * bound, bound)

                    file_x = '/BS/garvita/nobackup/pennaction_flownet/' + '{:05}_x.png'.format(i)
                    file_y = '/BS/garvita/nobackup/pennaction_flownet/' + '{:05}_y.png'.format(i)
                    cv2.imwrite(file_x, x_flow)
                    cv2.imwrite(file_y, y_flow)
                """
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
        Flows = video_flow[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X)
        flow = self.preprocess_img(Flows, flipcrop=True)
        #print(flow.shape)
        return flow

class PreprocessedDataset2(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, pose_dir, flow_dir, crop_size=(64, 64), video_len=32):
        self.dataset = dataset
        self.flow_dir = flow_dir
        self.pose_dir = pose_dir
        self.crop_x = crop_size[0]
        self.crop_y = crop_size[1]
        self.video_len = video_len
        self.stride = self.video_len//2

    def __len__(self):
        return len(self.dataset)

    def preprocess_img(self, flow, x_data, y_data, visibility, flipcrop=True):
        flipcrop=False
        crop_x = self.crop_x
        crop_y = self.crop_y
         #2,batchsize,img_dim
        #print(x_data, y_data, visibility)
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
            dst_flow = np.zeros((3,flow.shape[1],self.crop_y, self.crop_x), np.uint8)
            for i in range(flow.shape[1]):
                check2 = np.zeros((self.crop_x, self.crop_y), np.uint8)
                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[2, i] = scipy.misc.imresize(check2, [self.crop_y, self.crop_x], 'bicubic')
            flow = dst_flow
        else:
            dst_flow = np.zeros((3,flow.shape[1],self.crop_y, self.crop_x), np.uint8)
            check2 = np.zeros((self.crop_y, self.crop_x),  np.uint8)
            for i in range(flow.shape[1]):
                #ipdb.set_trace()
                for j in range(visibility.shape[1]):
                    #if(visibility[i][j] != 0):
                    if(y_data[i,j] < 64 & x_data[i][j] < 64):
                        check2[y_data[i,j], x_data[i,j]] = 255;
                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[2, i] = scipy.misc.imresize(check2, [self.crop_y, self.crop_x], 'bicubic')
            flow = dst_flow
        flow = flow.astype(np.float32) * (2 / 255.) - 1.
        return flow

    def get_example(self, i, train=True):
        while 1:
            if os.path.exists(self.flow_dir + self.dataset[i] + '.npy'):
                video_flow = np.load(self.flow_dir + self.dataset[i] + '.npy')
                N = video_flow.shape[0]
                pose_data = loadmat(self.pose_dir + self.dataset[i] + '.mat')
                x_data = np.array(pose_data['x']*self.crop_x/pose_data['dimensions'][0][1],np.int32)
                y_data = np.array(pose_data['y']*self.crop_y/pose_data['dimensions'][0][0],np.int32)
                visibility = pose_data['visibility']
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
        Flows = video_flow[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X)
        # above line changes (32, 76, 76, 2) to (2, 32, 76, 76) , replace 76 with 64 in pose case.
        #Poses = np.zeros((self.video_len, 64,64))
        X_data = x_data[j:j+self.video_len]
        Y_data = y_data[j:j+self.video_len]
        Visibilty = visibility[j:j+self.video_len]
        flow = self.preprocess_img(Flows,X_data, Y_data, visibility, flipcrop=False)
        return flow



