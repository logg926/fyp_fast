import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
from X2Facemaster.UnwrapMosaic.UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale

def run_batch(source_images, pose_images):
    return model(pose_images, *source_images)

BASE_MODEL = 'X2Facemaster/release_models/' # Change to your path
state_dict = torch.load(BASE_MODEL + 'x2face_model_forpython3.pth', map_location=torch.device('cpu'))

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
# model = model.cuda()

model = model.eval()


def load_img_from_array(array):
    img = Image.fromarray(array)
    transform = Compose([Scale((256,256)), ToTensor()])
    return Variable(transform(img))
    # return Variable(transform(img)).cuda()

# Driving the source image with the driving sequence

# img = Image.open(source_imgs[0])
# img = np.array(img)


def generateX2face(img, driver_imgs_array):
    # print(driver_imgs_array)
    source_images = []
    # for img in source_imgs:
    #     source_images.append(load_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))
    source_images.append(load_img_from_array(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))
    driver_images = None
    for img in driver_imgs_array:
        if driver_images is None:
            driver_images = load_img_from_array(img).unsqueeze(0)
        else:
            driver_images = torch.cat((driver_images, load_img_from_array(img).unsqueeze(0)), 0)
    # Run the model for each
    with torch.no_grad():
        result = run_batch(source_images, driver_images)
    result = result.clamp(min=0, max=1)
    # Run the model for each
    with torch.no_grad():
        result = run_batch(source_images, driver_images)

    result = result.clamp(min=0, max=1)

    output = []
    for frame in result.cpu().data:
        img = torchvision.utils.make_grid(frame)
        result_image= img.permute(1,2,0).numpy()
        output.append(result_image)
    return output


if __name__ == "__main__":
    driver_path = './examples/Taylor_Swift/1.6/nuBaabkzzzI/'
    source_path = './examples/Taylor_Swift/1.6/vBgiDYBCuxY/'

    driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))][0:4] # 8 driving frames
    # source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:3] # 3 source frames
    source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:1] # 3 source frames

    img = Image.open(source_imgs[0])

    driver_imgs_array = []
    for path in driver_imgs:
        driver_imgs_array.append(np.array(Image.open(path)))
    print(driver_imgs_array)
    img = np.array(img)

    # print("input")
    # print(img)
    # print("output")
    output = generateX2face(img, driver_imgs_array)
    print(output)
# plt.imshow(output[len(output)-1])