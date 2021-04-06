"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

NicolÃƒÂ² Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
# install dependencies if needed 
import os
os.system('pip install efficientnet_pytorch')
os.system('pip install albumentations==0.4.6')

import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit

# import sys
# sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from blazeface.read_video import read_frames_new
from architectures import fornet,weights
from isplutils import utils
import cv2

os.system('pwd')

# all the models architecture
models_set = ['Xception', 'EfficientNetB4', 'EfficientNetB4ST', 'EfficientNetAutoAttB4', 'EfficientNetAutoAttB4ST']
# all the models path
models_path = ['weights/Xception/net-Xception_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth', 
'weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41_nonTriplet/bestval.pth', 
'weights/binclass/net-EfficientNetB4ST_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth', 
'weights/binclass/net-EfficientNetAutoAttB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41_nonTriplet/bestval.pth',
'weights/binclass/net-EfficientNetAutoAttB4ST_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth']

# model variables 
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

# pre-load the models into device
models = []
for m, path in zip(models_set, models_path):
    try:
        with open(path, 'rb') as f:
            net = torch.load(f) if torch.cuda.is_available() else torch.load(f, map_location=torch.device('cpu'))
            # print(type(net['net']))
            model = getattr(fornet, m)().eval().to(device)
            model.load_state_dict(net['net'])
            models.append(model)
    except:
        # print('cannot load', path)
        models.append('')

# load face crop library
facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)

# load transformer
Xception_transf = utils.get_transformer(face_policy, face_size, models[0].get_normalizer(), train=False) if models[0] else ''
transf = utils.get_transformer(face_policy, face_size, models[1].get_normalizer(), train=False) if models[1] else ''


# def process_videos(self, input_dir, filenames, video_idxs):
#     target_size = self.facedet.input_size

#     videos_read = []
#     frames_read = []
#     frames = []
#     tiles = []
#     resize_info = []

#     for video_idx in video_idxs:
#         # Read the full-size frames from this video.
#         filename = filenames[video_idx]
#         video_path = os.path.join(input_dir, filename)
#         result = self.video_read_fn(video_path)

#         # Error? Then skip this video.
#         if result is None: continue

#         videos_read.append(video_idx)

#         # Keep track of the original frames (need them later).
#         my_frames, my_idxs = result
#         frames.append(my_frames)
#         frames_read.append(my_idxs)

#         # Split the frames into several tiles. Resize the tiles to 128x128.
#         my_tiles, my_resize_info = self._tile_frames(my_frames, target_size)
#         tiles.append(my_tiles)
#         resize_info.append(my_resize_info)

#     if len(tiles) == 0:
#         return []
#     # Put all the tiles for all the frames from all the videos into
#     # a single batch.
#     batch = np.concatenate(tiles)

#     # Run the face detector. The result is a list of PyTorch tensors,
#     # one for each image in the batch.
#     all_detections = self.facedet.predict_on_batch(batch, apply_nms=False)

#     result = []
#     offs = 0
#     for v in range(len(tiles)):
#         # Not all videos may have the same number of tiles, so find which
#         # detections go with which video.
#         num_tiles = tiles[v].shape[0]
#         detections = all_detections[offs:offs + num_tiles]
#         offs += num_tiles

#         # Convert the detections from 128x128 back to the original frame size.
#         detections = self._resize_detections(detections, target_size, resize_info[v])

#         # Because we have several tiles for each frame, combine the predictions
#         # from these tiles. The result is a list of PyTorch tensors, but now one
#         # for each frame (rather than each tile).
#         num_frames = frames[v].shape[0]
#         frame_size = (frames[v].shape[2], frames[v].shape[1])
#         detections = self._untile_detections(num_frames, frame_size, detections)

#         # The same face may have been detected in multiple tiles, so filter out
#         # overlapping detections. This is done separately for each frame.
#         detections = self.facedet.nms(detections)

#         for i in range(len(detections)):
#             # Crop the faces out of the original frame.
#             frameref_detections = self._add_margin_to_detections(detections[i], frame_size, 0.2)
#             faces = self._crop_faces(frames[v][i], frameref_detections)
#             kpts = self._crop_kpts(frames[v][i], detections[i], 0.3)

#             # Add additional information about the frame and detections.
#             scores = list(detections[i][:, 16].cpu().numpy())
#             frame_dict = {"video_idx": videos_read[v],
#                             "frame_idx": frames_read[v][i],
#                             "frame_w": frame_size[0],
#                             "frame_h": frame_size[1],
#                             "frame": frames[v][i],
#                             "faces": faces,
#                             "kpts": kpts,
#                             "detections": frameref_detections.cpu().numpy(),
#                             "scores": scores,
#                             }
#             # Sort faces by descending confidence
#             frame_dict = self._soft_faces_by_descending_score(frame_dict)

#             result.append(frame_dict)

#     return result


def predict(pathToVid, testOnModels=[]):
    faces = ''
    input_dir = os.path.dirname(pathToVid)
    filenames = [os.path.basename(pathToVid)]
    # faces = process_videos(input_dir, filenames, [0])

    filename = filenames[0]
    video_path = os.path.join(input_dir, filename)
    # result = video_read_fn(video_path)

    # capture = cv2.VideoCapture(video_path)
    # print(capture)
    # result = videoreader.read_frames(video_path, num_frames=frames_per_video)

    result = read_frames_new(video_path, num_frames=frames_per_video)
    print(result) 
    faces = face_extractor.process_videos(input_dir, filenames, facedet, video_read_fn, result)
    # print (faces)
    # faces = face_extractor.process_video(pathToVid)
    
    scores = []
    for m in testOnModels:
        if m not in models_set:
            print('models ' + m + 'is not defined')
        index = models_set.index(m)
        if index > 0:
            faces_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces']) ] )
        else:
            faces_t = torch.stack( [ Xception_transf(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces']) ] )
        with torch.no_grad():
            if models[index]:
                pred = models[index](faces_t.to(device)).cpu().numpy().flatten()
                # sigmoid function for finding score decision
                scores.append(expit(pred.mean()))
            else: print('models ' + m + 'is empty')
    # score closer to 0 means its real, closer to 1 means fake
    avg_score = sum(scores) / len(scores) if scores else -1
    return 'Real' if avg_score < 0.5 else 'Fake', avg_score



def predict_cnn(result, testOnModels=[]):
    # faces = ''
    # input_dir = os.path.dirname(pathToVid)
    # filenames = [os.path.basename(pathToVid)]
    # # faces = process_videos(input_dir, filenames, [0])

    # filename = filenames[0]
    # video_path = os.path.join(input_dir, filename)
    # # result = video_read_fn(video_path)

    # # capture = cv2.VideoCapture(video_path)
    # # print(capture)
    # # result = videoreader.read_frames(video_path, num_frames=frames_per_video)

    # result = read_frames_new(video_path, num_frames=frames_per_video)
    # print(result) 
    faces = face_extractor.process_videos(input_dir, filenames, facedet, video_read_fn, result)
    # print (faces)
    # faces = face_extractor.process_video(pathToVid)
    
    scores = []
    for m in testOnModels:
        if m not in models_set:
            print('models ' + m + 'is not defined')
        index = models_set.index(m)
        if index > 0:
            faces_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces']) ] )
        else:
            faces_t = torch.stack( [ Xception_transf(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces']) ] )
        with torch.no_grad():
            if models[index]:
                pred = models[index](faces_t.to(device)).cpu().numpy().flatten()
                # sigmoid function for finding score decision
                scores.append(expit(pred.mean()))
            else: print('models ' + m + 'is empty')
    # score closer to 0 means its real, closer to 1 means fake
    avg_score = sum(scores) / len(scores) if scores else -1
    return 'Real' if avg_score < 0.5 else 'Fake', avg_score

if __name__ == '__main__':
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str)
    # parser.add_argument('--modelsToEval', type=str)
    # args = parser.parse_args()
    # modelsToEval = args.modelsToEval.split(',')
    # print('Evaluating models:', modelsToEval)
    # print(predict(args.path, modelsToEval))
    print(predict("../test_dataset/real/sqqamveljk.mp4", ["Xception"]))
