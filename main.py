from typing import Optional

from fastapi import FastAPI
from generate import generateNew 
import imageio
import numpy as np
app = FastAPI()


@app.get("/")
def read_root():
    source_image = np.array(imageio.imread('./erik.jpeg'))
    # print(source_image)
    reader = imageio.get_reader('./damedaneshort.mp4')
    fps = reader.get_meta_data()['fps']
    drive_vid = []
    try:
        for im in reader:
            drive_vid.append(im)
    except RuntimeError:
        print('fail to read drive video from source ' + driving_video)
        pass
    reader.close()
    # print(drive_vid)
    generateNew(source_image, drive_vid, './result_erik.mp4', fps)

    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}