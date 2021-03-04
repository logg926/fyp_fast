from generate import generateNew 
import imageio
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
import dataclasses
import json
from pydantic.dataclasses import dataclass
import cv2
import pandas as pd
# source_image = np.asarray(cv2.imread('./erik.jpeg').tolist())
import json



source_image = np.array(cv2.imread('./erik.jpeg'))

print(source_image.dtype)

source_image = np.array(source_image.tolist(), dtype = "uint8")
print(source_image.shape)
# cv2.imwrite('output.jpg', source_image)
reader = imageio.get_reader('./damedaneshort.mp4')
fps = reader.get_meta_data()['fps']
# print(fps) # 24
drive_vid = []
try:
    for im in reader:
        drive_vid.append(im)
except RuntimeError:
    print('fail to read drive video from source ' + driving_video)
    pass
reader.close()
# with open('test.json', 'w') as f:
#     json.dump({ "sourceimg": source_image, "drivevid": drive_vid, "fps": 24}, f, cls=NumpyEncoder)
# # print(drive_vid)

# print(drive_vid.shape)
generateNew(source_image, drive_vid, './result_erik7.mp4', fps)