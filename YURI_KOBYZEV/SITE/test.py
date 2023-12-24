import numpy as np
import os

os.environ['USE_TORCH'] = '1'

from doctr.io import DocumentFile
from doctr.models import detection_predictor

model = detection_predictor(arch='db_resnet50', pretrained=True)
input_page = DocumentFile.from_images("753_1.jpg")
#input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)

out = model(input_page)
print(out)

