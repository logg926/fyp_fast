

import uuid
from starlette.staticfiles import StaticFiles
from fastapi import FastAPI, Form
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
from ensemblecnn.predict import predict_cnn
from X2Facemaster.UnwrapMosaic.generate import generateX2face
import pickle
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

SVM = pickle.load(open('./SVM model_v0.24.1.pkl', 'rb'))
size = 300
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
    source_image = np.array(item.sourceimg, dtype="uint8")
    drive_vid = np.asarray(item.drivevid, dtype="uint8")
    imageio.imsave('./'+"testing2.jpg", source_image)
    fps = item.fps
    result = generateNew(source_image, drive_vid, './heretemp.mp4', fps)
    response = []
    for res in result:
        response.append(res.tolist())

    filename = str(uuid.uuid4())
    url = 'static/'+filename+'.mp4'
    imageio.mimsave('./'+url, result, fps=fps)
    return url


@app.post("/svm_test")
def svm_testf(item: TestObj):
    img = np.array(item.detectimg, dtype="uint8")
    feature = transformFrame(img, size)
    prediction = SVM.predict(np.array([feature]))
    return JSONResponse(content=json.dumps(prediction.tolist()))


@app.post("/capsule_test")
def svm_test(item: TestCapObj):
    vid = np.array(item.detectvid, dtype="uint8")
    cls, prob = detect(vid)
    return JSONResponse(content=json.dumps(prob))


@app.post("/cnn_test")
def svm_tesdt(item: TestCapObj):
    vid = np.array(item.detectvid, dtype="uint8")

    filename = str(uuid.uuid4())
    place = './tempvid/'+filename+'.mp4'
    imageio.mimsave(place, vid, fps=15)

    prob = predict_cnn(place, ["Xception"])
    return JSONResponse(content=json.dumps(prob))


@app.post("/ensemble_test")
def svm_tesdte(item: TestCapObj):
    vid = np.array(item.detectvid, dtype="uint8")

    filename = str(uuid.uuid4())
    place = './tempvid/'+filename+'.mp4'
    imageio.mimsave(place, vid, fps=15)

    prob = predict_cnn(place, ['Xception', 'EfficientNetB4', 'EfficientNetB4ST',
                'EfficientNetAutoAttB4', 'EfficientNetAutoAttB4ST'])
    return JSONResponse(content=json.dumps(prob))


@app.post("/x2gen")
def x2gend(item: GenearateObj):
    source_image = np.array(item.sourceimg, dtype="uint8")
    drive_vid = np.asarray(item.drivevid, dtype="uint8")
    fps = item.fps
    driver_imgs = drive_vid
    result = generateX2face(source_image, driver_imgs)
    response = []
    for res in result:
        response.append(res.tolist())
    filename = str(uuid.uuid4())
    url = 'static/'+filename+'.mp4'
    imageio.mimsave('./'+url, result, fps=fps)
    return url


@app.get("/items/{item_id}")
def new(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


app.mount("/static", StaticFiles(directory="static"))
