from paddleocr import PaddleOCR
import transformations as tr
import os
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
ocr = PaddleOCR(use_angle_cls=True,use_gpu=False,lang='en',det_db_thresh=0.05, det_db_box_thresh=0.1)
tra = tr.Aligner()
img_path = './images/voice0002-2.jpg'
img = cv2.imread(img_path)
img = tra.cropping_image_by_markers(img)
result = ocr.ocr(img)
res = result[0]
for i,r in enumerate(res):
    print("line:",i,"box:",r[0], " value",r[1][0], " prob:",r[1][1])
