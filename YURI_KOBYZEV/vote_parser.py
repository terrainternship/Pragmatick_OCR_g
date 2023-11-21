import os
os.environ['USE_TORCH'] = '1'
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image, ImageReadMode


device = "cuda" if torch.cuda.is_available() else "cpu"



vmodel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
vmodel.fc = torch.nn.Linear(2048, 5)
vmodel.to(device)

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
model.to(device)

# PDF
# doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# doc = DocumentFile.from_images("753_1.jpg")
# doc = DocumentFile.from_images("753_2.jpg")
doc = DocumentFile.from_images("753_3.jpg")
#doc=doc.to(device)

# Analyze
result = model(doc)
data = result.export()
line=-1
for p in data['pages']:
    for b in p['blocks']:
        line+=1
        #print('line:',line)
        for l in b['lines']:
            for w in l['words']: 
                if w['value']=='3A':
                #    print('w:',w)
                    za = f"0, {w['geometry'][0][1]:.3f}, 1.0, {w['geometry'][1][1]:.3f}"
                    print("ZA:",za) # можно вызывать тут предикт результата голосования: model1.predict(za)
                if w['value'].startswith('N'):
                    lenval=len(w['value'])
                    n=w['value'][2:lenval]
                    if lenval>=28:
                        print("doc N:",n)
                        if line>3: 
                            # print("prevline:",prevline)
                            listnum=prevline['value']
                            print("list N:",listnum)
                if w['value']=='@MO':
                    sign = f"0, {prevline['geometry'][0][1]:.3f}, 1.0, {prevline['geometry'][1][1]:.3f}"
                    print("Sign:",sign) # можно вызывать тут предикт результата подписи : model2.predict(sign)
                prevline=w
