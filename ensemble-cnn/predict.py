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
from architectures import fornet,weights
from isplutils import utils

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

def predict(pathToVid, testOnModels=[]):
    faces = ''
    try:
        faces = face_extractor.process_video(pathToVid)
    except:
        print('error in face extractor')
    
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
