

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from generate import generateNew 
from testfrequencydomain import transformFrame
import imageio
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from CapsuleForensicsv2.test_vid_function import detect
import dataclasses
import json
from pydantic.dataclasses import dataclass
import cv2

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ensemble-cnn.predict import predict_cnn

import pickle
app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # Allows all origins
allow_credentials=True,
allow_methods=["*"], # Allows all methods
allow_headers=["*"], # Allows all headers
)
print ("hello")
# = dataclasses.field(default_factory=lambda: [[[[0]]]])

SVM = pickle.load(open('./SVM model_v0.24.1.pkl', 'rb'))
size = 300
@app.get("/")
def read_root():
    # source_image = np.array(imageio.imread('./erik.jpeg'))
    # print(source_image.shape)
    # source_image = cv2.imread('./erik.jpeg')
    # print(source_image)
    # # cv2.imwrite('output.jpg', source_image)
    # reader = imageio.get_reader('./damedaneshort.mp4')
    # fps = reader.get_meta_data()['fps']
    # print(fps) # 24
    # drive_vid = []
    # try:
    #     for im in reader:
    #         drive_vid.append(im)
    # except RuntimeError:
    #     print('fail to read drive video from source ' + driving_video)
    #     pass
    # reader.close()
    # # with open('test.json', 'w') as f:
    # #     json.dump({ "sourceimg": source_image, "drivevid": drive_vid, "fps": 24}, f, cls=NumpyEncoder)
    # # # print(drive_vid)
    # # print(drive_vid.shape)
    # generateNew(source_image, drive_vid, './result_erik6.mp4', fps)
    return JSONResponse(content=json.dumps([[[[0]]]]))



class GenearateObj(BaseModel):
    sourceimg: List[List[List[int]]] 
    drivevid: List[List[List[List[int]]]] 
    fps: int

class TestObj(BaseModel):
    detectimg: List[List[int]]

class TestCapObj(BaseModel):
    detectvid: List[List[List[List[int]]]] 

@app.post("/gen")
def read_item(item: GenearateObj):
    source_image = np.array(item.sourceimg, dtype = "uint8")
    drive_vid = np.asarray(item.drivevid, dtype = "uint8")
    fps = item.fps
    result = generateNew(source_image, drive_vid, './result_erik5.mp4', fps)
    print(type(result[0]))
    response = []
    for res in result:
        response.append(res.tolist())
    # imageio.mimsave('./result_erik5.mp4', result, fps=fps)
    return JSONResponse(content=json.dumps(response))

@app.post("/svm_test")
def svm_testf(item: TestObj):
    img = np.array(item.detectimg, dtype = "uint8")
    feature = transformFrame(img, size)
    prediction = SVM.predict(np.array([feature]))
    return JSONResponse(content=json.dumps(prediction.tolist()))

@app.post("/capsule_test")
def svm_test(item: TestCapObj):
    # vid = imageio.get_reader('./test_dataset/real/sqqamveljk.mp4', fps=5)
    # frames = []
    # reader = vid
    # try:
    #     for im in reader:
    #         frames.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()
    vid = np.array(item.detectvid, dtype = "uint8")
    cls, prob = detect(vid)
    return JSONResponse(content=json.dumps(prob))

@app.post("/cnn_test")
def svm_tesdt(item: TestCapObj):
    # vid = imageio.get_reader('./test_dataset/real/sqqamveljk.mp4', fps=5)
    # frames = []
    # reader = vid
    # try:
    #     for im in reader:
    #         frames.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()
    vid = np.array(item.detectvid, dtype = "uint8")
    predict_cnn(vid, ["Xception"])
    return JSONResponse(content=json.dumps(prob))

@app.post("/x2gen")
def x2gend(item: GenearateObj):
    source_image = np.array(item.sourceimg, dtype = "uint8")
    drive_vid = np.asarray(item.drivevid, dtype = "uint8")
    fps = item.fps
    result = generateX2face(source_image, drive_vid)
    # print(type(result[0]))
    response = []
    for res in result:
        response.append(res.tolist())
    # imageio.mimsave('./result_erik5.mp4', result, fps=fps)
    return JSONResponse(content=json.dumps(response))

@app.get("/items/{item_id}")
def new(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


