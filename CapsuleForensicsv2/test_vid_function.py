"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on videos level by aggregating the predicted probabilities of their frames using FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from PIL import Image
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import math
import model_big
import imageio
from skimage.transform import resize
# os.system("pip uninstall face_recognition")
# import dlib
# dlib.DLIB_USE_CUDA = False
import face_recognition

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp/test', help='path to test dataset')
parser.add_argument('--real', default ='0_original', help='real folder name')
parser.add_argument('--deepfakes', default ='1_deepfakes', help='fake folder name')
parser.add_argument('--face2face', default ='2_face2face', help='fake folder name')
parser.add_argument('--faceswap', default ='3_faceswap', help='fake folder name')
parser.add_argument('--batchSize', type=int, default=10, help='batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output images and model checkpoints')
parser.add_argument('--random_sample', type=int, default=0, help='number of random sample to test')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

img_ext_lst = ('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'gif', 'tiff')
vid_ext_lst = ('avi', 'mkv', 'mpeg', 'mpg', 'mp4')

def get_file_list(path, ext_lst):
    file_lst = []

    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if f.lower().endswith(ext_lst):
                file_lst.append(f)

    return file_lst

def extract_file_name_without_count(in_str, sep_char='_'):
    n = len(in_str)
    pos = 0
    for i in range(n):
        if in_str[i] == sep_char:
            pos = i

    return in_str[0:pos]

def process_file_list(file_lst, sep_char='_'):
    result_lst = []

    for i in range(len(file_lst)):
        #remove extension
        filename = os.path.splitext(file_lst[i])[0]

        filename = extract_file_name_without_count(filename, sep_char)
        result_lst.append(filename)

    return result_lst

def classify_batch(vgg_ext, model, batch):
    n_sub_imgs = len(batch)

    if (opt.random_sample > 0):
        if n_sub_imgs > opt.random_sample:
            np.random.shuffle(batch)
            n_sub_imgs = opt.random_sample

        img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

        for i in range(n_sub_imgs):
            img_tmp = torch.cat((img_tmp, batch[i]), dim=0)

        if opt.gpu_id > 0:
            img_tmp = img_tmp.cuda(opt.gpu_id)

        input_v = Variable(img_tmp, requires_grad = False)

        x = vgg_ext(input_v)
        classes, class_ = model(x, random=opt.random)
        output_pred = class_.data.cpu().numpy()

    else:
        batchSize = opt.batchSize
        steps = int(math.ceil(n_sub_imgs*1.0/batchSize))

        output_pred = np.array([], dtype=np.float).reshape(0,2)

        for i in range(steps):

            img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

            end = (i + 1)*batchSize
            if end > n_sub_imgs:
                end = n_sub_imgs - i*batchSize
            else:
                end = batchSize

            for j in range(end):
                img_tmp = torch.cat((img_tmp, batch[i*batchSize + j]), dim=0)

            if opt.gpu_id > 0:
                img_tmp = img_tmp.cuda(opt.gpu_id)

            input_v = Variable(img_tmp, requires_grad = False)

            x = vgg_ext(input_v)
            classes, class_ = model(x, random=opt.random)
            output_p = class_.data.cpu().numpy()

            output_pred = np.concatenate((output_pred, output_p), axis=0)

    output_pred = output_pred.mean(0)

    if output_pred[1] >= output_pred[0]:
        pred = 1
    else:
        pred = 0

    return pred, output_pred[1]

transform_fwd = transforms.Compose([
    transforms.Resize((opt.imageSize, opt.imageSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def cropFace(frame, model='hog'):
    face_locations = face_recognition.face_locations(frame, model=model)
    if len(face_locations) > 0:
        if len(face_locations) > 1:
            face_locations = sorted(face_locations, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]), reverse=True)[0]
        else: face_locations = face_locations[0]
        # crop square face for better resize performance
        # center in (row, col format)
        center = ((face_locations[0]+face_locations[2])//2, (face_locations[1]+face_locations[3])//2)
        return findRange(frame, 300, center)

def findRange(frame, dimension, faceCenter):
    if len(frame) < 300 or len(frame[0]) < 300: 
        # if the dimension of original frame < 300
        return resize(frame, (300, 300))[..., :3]
    offset = dimension//2
    startRow = 0 if faceCenter[0] < offset else faceCenter[0] - offset
    startCol = 0 if faceCenter[1] < offset else faceCenter[1] - offset
    return frame[startRow:startRow+dimension, startCol:startCol+dimension]

def classify_frames(vgg_ext, model, vid):
    
    # place holder logic for testing with detect dataset, reset to throw error in production
    
    # videoDir = './test_dataset/real/sqqamveljk.mp4'
    # videoDir = vid
    # reader = imageio.get_reader(videoDir, fps=5)
    # frames = []
    # reader = vid
    # try:
    #     for im in reader:
    #         frames.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()

    frames = vid

    frames = [transform_fwd(Image.fromarray(cropFace(f))).unsqueeze(0) for f in frames]
    return classify_batch(vgg_ext, model, frames)
    # if not vid:
    #     # place holder logic for testing with detect dataset, reset to throw error in production
    #     frames = []
    #     videoDir = './test_dataset/real/sqqamveljk.mp4'
    #     reader = imageio.get_reader(videoDir, fps=5)
    #     try:
    #         for im in reader:
    #             frames.append(im)
    #     except RuntimeError:
    #         pass
    #     reader.close()

    #     frames = [transform_fwd(Image.fromarray(cropFace(f))).unsqueeze(0) for f in frames]
    #     return classify_batch(vgg_ext, model, frames)
    # else:
    #     frames = [transform_fwd(Image.fromarray(cropFace(f))).unsqueeze(0) for f in vid]
    #     return classify_batch(vgg_ext, model, frames)

def detect(vid):
    vgg_ext = model_big.VggExtractor()
    model = model_big.CapsuleNet(2)
    GPU = -1
    MODEL_ID = 8
    MODEL_PATH = './CapsuleForensicsv2/checkpoints/binary_faceforensicspp_v2_full/capsule_' + str(MODEL_ID) +'.pt'

    if GPU < 0:
        # print('here')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    if GPU > 0:
        vgg_ext.cuda(GPU)
        model.cuda(GPU)


    cls, prob = classify_frames(vgg_ext, model, vid)
    
    return cls, prob

if __name__ == '__main__':
    vid = imageio.get_reader('./test_dataset/real/sqqamveljk.mp4', fps=5)
    frames = []
    reader = vid
    try:
        for im in reader:
            frames.append(im)
    except RuntimeError:
        pass
    reader.close()
    print(frames)
    cls, prob = detect(frames)
    print(('Real', prob) if cls == 0 else ('Fake', prob))