import random

import lmdb
import cv2 as cv
import numpy as np

data_path = r'D:\text_recoginition\train\font_3'
env = lmdb.open(data_path,
                max_readers=8,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
txn = env.begin(write=False)
nSample: int = int(txn.get('num-samples'.encode()))
print(nSample)

id = random.randint(0, nSample)

img_code: str = 'image-%09d' % id
imgbuf = txn.get(img_code.encode())
img = np.frombuffer(imgbuf, dtype=np.uint8)
img = cv.imdecode(img, cv.IMREAD_COLOR)
# img = img[10: -10, :, :]
# img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv.INTER_CUBIC)
print(img.shape)

label_code: str = 'label-%09d' % id
byte_label: bytes = txn.get(label_code.encode())
label = byte_label.decode("utf-8")
label = label.strip("\n").strip("\r\t")
print(label)
cv.imshow("abc", img)
cv.waitKey(0)
