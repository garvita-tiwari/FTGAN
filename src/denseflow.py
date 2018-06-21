import cv2
import numpy as np
import os
import scipy.misc
from PIL import Image
import random

bound = 20
parameter = 2
dataset = 'surreal'
save_dir_type = 'data_dense'
import numpy as np
import cv2

def CheckVideo(video):
    e = -1 if len(video)%2 else -2
    tmp = video[0:e:2] - video[1:-1:2]
    tmp = tmp.sum(axis=3).sum(axis=2).sum(axis=1).astype(np.bool)
    ratio = 1 - tmp.mean()
    if ratio > 0.2:
        return video[0::2], ratio
    else:
        return video, ratio

def CalcFlow(video, parameter = 2):
    for i in range(0, len(video) - 1):
        prev = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(video[i + 1], cv2.COLOR_BGR2GRAY)

        if parameter == 1:
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.702, 5, 10, 2, 7, 1.5,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        elif parameter == 2:
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif parameter == 3:
            a=1
            #flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 3, 3, 5, 1.1, 0)
        if i == 0:
            Flows = np.zeros((len(video) - 1,) + flow.shape, flow.dtype)
        Flows[i] = flow
    return Flows

def ConvertFlow2Img(Flows, lower_bound, higher_bound):
    DstFlows = (Flows - lower_bound) / float(higher_bound - lower_bound) * 255.
    DstFlows = np.round(DstFlows).astype(np.uint8)

    low_index = np.where(Flows < lower_bound)
    DstFlows[low_index] = 0
    high_index = np.where(Flows > higher_bound)
    DstFlows[high_index] = 255

    return DstFlows

if dataset == 'surreal':
    m_cap = 'cmu'
    data_root = '/data/unagi0/dataset/surreal/SURREAL/data/' + m_cap + '/'
    frame_root = '/data/unagi0/ohnishi/video_generation/human/data/surreal/' + m_cap + '/frames/'
    npy_root = '/data/unagi0/ohnishi/video_generation/human/data/surreal/' + m_cap + '/npy/'
    save_root_flow = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/surreal/' + m_cap + '/npy_flow/'
    save_root_small_img = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/surreal/' + m_cap + '/npy_76/'
    save_root_small_flow = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/surreal/' + m_cap + '/npy_flow_76/'

    for data_type in ['train', 'test']:
        Data = []
        f = open(
            '/home/mil/ohnishi/workspace/video_generation/videogeneration_v2/data/surreal/' + data_type + '_modified.txt')
        for line in f.readlines():
            Data.append(line.split()[0])
        f.close()

        random.shuffle(Data)

        for data in Data:
            npy_path = npy_root + data_type + '/' + data + '.npy'
            save_path_flow = npy_path.replace(npy_root, save_root_flow)
            if not os.path.exists(save_path_flow):
                print save_path_flow
                video = np.load(npy_path)

                Flows = CalcFlow(video, parameter=parameter)
                Flows = ConvertFlow2Img(Flows, -1 * bound, bound)

                SmallImgs = np.zeros((video.shape[0], 76, 76, 3), np.uint8)
                SmallFlows = np.zeros((video.shape[0] - 1, 76, 76, 2), np.uint8)
                for i in range(video.shape[0]):
                    SmallImgs[i] = scipy.misc.imresize(video[i], [76, 76], 'bicubic')
                    if i < video.shape[0] - 1:
                        SmallFlows[i, :, :, 0] = scipy.misc.imresize(Flows[i, :, :, 0], [76, 76], 'bicubic')
                        SmallFlows[i, :, :, 1] = scipy.misc.imresize(Flows[i, :, :, 1], [76, 76], 'bicubic')

                save_path_small_flow = save_path_flow.replace(save_root_flow, save_root_small_flow)
                save_dir = os.path.split(save_path_small_flow)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path_small_flow, SmallFlows)

                save_path_small_img = npy_path.replace(npy_root, save_root_small_img)
                save_dir = os.path.split(save_path_small_img)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path_small_img, SmallImgs)

                save_dir = os.path.split(save_path_flow)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path_flow, Flows)


elif dataset == 'penn_action':
    npy_root = '/data/unagi0/ohnishi/video_generation/human/data/Penn_Action/npy/'
    save_root_flow = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/Penn_Action/npy_flow/'
    save_root_small_img = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/Penn_Action/npy_76/'
    save_root_small_flow = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/Penn_Action/npy_flow_76/'

    VideoIds = range(1, 2327)

    # random.shuffle(VideoIds)
    for video_id in VideoIds:
        npy_path = npy_root + '{:04}.npy'.format(video_id)

        save_path_flow = npy_path.replace(npy_root, save_root_flow)
        if not os.path.exists(save_path_flow):
            video = np.load(npy_path)

            video, ratio = CheckVideo(video)

            print 'ratio:', ratio, '\t video_id:', video_id
            Flows = CalcFlow(video, parameter=parameter)
            Flows = ConvertFlow2Img(Flows, -1 * bound, bound)

            SmallImgs = np.zeros((video.shape[0], 76, 76, 3), np.uint8)
            SmallFlows = np.zeros((video.shape[0] - 1, 76, 76, 2), np.uint8)
            for i in range(video.shape[0]):
                SmallImgs[i] = scipy.misc.imresize(video[i], [76, 76], 'bicubic')
                if i < video.shape[0] - 1:
                    SmallFlows[i, :, :, 0] = scipy.misc.imresize(Flows[i, :, :, 0], [76, 76], 'bicubic')
                    SmallFlows[i, :, :, 1] = scipy.misc.imresize(Flows[i, :, :, 1], [76, 76], 'bicubic')

            save_path_small_flow = save_path_flow.replace(save_root_flow, save_root_small_flow)
            save_dir = os.path.split(save_path_small_flow)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_path_small_flow, SmallFlows)

            save_path_small_img = npy_path.replace(npy_root, save_root_small_img)
            save_dir = os.path.split(save_path_small_img)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_path_small_img, SmallImgs)

            save_dir = os.path.split(save_path_flow)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_path_flow, Flows)
