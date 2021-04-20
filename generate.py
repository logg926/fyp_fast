import matplotlib
matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

# from pytorch2keras.converter import pytorch_to_keras
# imageio.plugins.ffmpeg.download()
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

def generate(source_image, driving_video, result_video):
    CONFIG = 'config/vox-256.yaml'
    CHECKPOINT = './vox-cpk.pth.tar'
    RELATIVE = False
    ADAPT_SCALE = False
    FIND_BEST_FRAME = False
    CPU = True

    source_image = imageio.imread(source_image)
    reader = imageio.get_reader(driving_video)
    fps = reader.get_meta_data()['fps']
    drive_vid = []
    try:
        for im in reader:
            drive_vid.append(im)
    except RuntimeError:
        print('fail to read drive video from source ' + driving_video)
        pass
    reader.close()

    source_img = resize(source_image, (256, 256))[..., :3]
    driving_vid = [resize(frame, (256, 256))[..., :3] for frame in drive_vid]
    generator, kp_detector = load_checkpoints(config_path=CONFIG, checkpoint_path=CHECKPOINT, cpu=CPU)
    if FIND_BEST_FRAME:
        i = find_best_frame(source_img, driving_vid, cpu=CPU)
        print ("Best frame: " + str(i))
        driving_forward = driving_vid[i:]
        driving_backward = driving_vid[:(i+1)][::-1]
        predictions_forward = make_animation(source_img, driving_forward, generator, kp_detector, relative=RELATIVE, adapt_movement_scale=ADAPT_SCALE, cpu=CPU)
        predictions_backward = make_animation(source_img, driving_backward, generator, kp_detector, relative=RELATIVE, adapt_movement_scale=ADAPT_SCALE, cpu=CPU)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_img, driving_vid, generator, kp_detector, relative=RELATIVE, adapt_movement_scale=ADAPT_SCALE, cpu=CPU)
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)




def generateNew(source_image, drive_vid, result_video, fps):
    CONFIG = 'config/vox-256.yaml'
    CHECKPOINT = './vox-cpk.pth.tar'
    RELATIVE = False
    ADAPT_SCALE = False
    FIND_BEST_FRAME = False
    CPU = True
    source_img = resize(source_image, (256, 256))[..., :3]
    driving_vid = [resize(frame, (256, 256))[..., :3] for frame in drive_vid]
    generator, kp_detector = load_checkpoints(config_path=CONFIG, checkpoint_path=CHECKPOINT, cpu=CPU)

    # source = torch.tensor(source_img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    # driving = torch.tensor(np.array(driving_vid)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    # driving_frame = driving[:, :, 0]
    # torch.onnx.export(generator, (source,kp_detector(driving_frame),  kp_detector(source)), 'onnx_model_generator.onnx', verbose=True)
    if FIND_BEST_FRAME:
        i = find_best_frame(source_img, driving_vid, cpu=CPU)
        print ("Best frame: " + str(i))
        driving_forward = driving_vid[i:]
        driving_backward = driving_vid[:(i+1)][::-1]
        predictions_forward = make_animation(source_img, driving_forward, generator, kp_detector, relative=RELATIVE, adapt_movement_scale=ADAPT_SCALE, cpu=CPU)
        predictions_backward = make_animation(source_img, driving_backward, generator, kp_detector, relative=RELATIVE, adapt_movement_scale=ADAPT_SCALE, cpu=CPU)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_img, driving_vid, generator, kp_detector, relative=RELATIVE, adapt_movement_scale=ADAPT_SCALE, cpu=CPU)
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    return [img_as_ubyte(frame) for frame in predictions]

# if __name__ == "__main__":
#     source_image = numpy.array(imageio.imread('./erik.jpeg'))
#     # print(source_image)
#     reader = imageio.get_reader('./damedaneshort.mp4')
#     fps = reader.get_meta_data()['fps']
#     drive_vid = []
#     try:
#         for im in reader:
#             drive_vid.append(im)
#     except RuntimeError:
#         print('fail to read drive video from source ' + driving_video)
#         pass
#     reader.close()
#     # print(drive_vid)
#     generateNew(source_image, drive_vid, './result_erik.mp4', fps)
