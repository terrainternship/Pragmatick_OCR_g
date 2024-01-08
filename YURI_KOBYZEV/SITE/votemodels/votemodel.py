import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import resnet50,ResNet50_Weights
from PIL import Image

def reslabel(a):
  r=''.join([str(s) for s in a.astype(int).tolist()])
  return  r



class Resnext50ml(torch.nn.Module):
    def __init__(self, vpath,device,n_classes):
        super().__init__()
        print("init model Resnext50ml: ",vpath,n_classes,device)
        resnet = models.resnext50_32x4d()
        resnet.fc = torch.nn.Sequential(
            nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = torch.nn.Sigmoid() # for multi label we need sigmoid! not crossentropy
        self.load_state_dict(torch.load(vpath,map_location=device))
        self.i2l = {'100': 'ЗА',
                    '010': 'ПРОТИВ',
                    '001': 'ВОЗДЕРЖАЛСЯ',
                    '000': 'НЕГОЛОСОВАЛ',
                    '110': 'И-ЗП',
                    '101': "И-ЗВ",
                    '011': 'И-ПВ',
                    '111': 'И-ЗПВ', 
                    }

        # inference mode
        self.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.device=device
        self.transform = transforms.Compose([ 
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def forward(self, x):
        return self.sigm(self.base_model(x))

    def preprocess(self, img, xmin, ymin, xmax, ymax): 
        (h,w,c) = img.shape
        za = [xmin*w, ymin*h, xmax*w, ymax*h]
        im = Image.fromarray(img).crop(za)
        tim = self.transform(im)
        tim = tim.unsqueeze(0)
        return(tim)

    def predict(self,im,thr):
        with torch.no_grad(): 
            im.to(self.device)
            raw_prob = self.forward(im.float()).detach().numpy()[0] 
            if thr==None:
                thr=0.5
            raw_pred = np.array(raw_prob > thr, dtype=int) 
            pred_lab = self.i2l[reslabel(raw_pred)]
            #mlabel=str(raw_pred[0])+str(raw_pred[1])+str(raw_pred[2])+str(raw_pred[3])
            mlabel=''.join([str(r) for r in raw_pred])
            rp = np.round(raw_prob,decimals=3)
            r = [str(x) for x in rp]
            #r = [str(rp[0]),str(rp[1]),str(rp[2]),str(rp[3])]
            print('r rounded list:',r)
            return {'label':  pred_lab, 'mlabel': mlabel, 'prob': r}

################################

class Resnet50s(torch.nn.Module):
    def __init__(self,mpath,device,n_classes=1):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = torch.nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes))
        resnet.load_state_dict(torch.load(mpath,map_location=device))
        self.base_model = resnet
        # inference mode
        self.eval()
        self.device=device
        self.transform = transforms.Compose([ 
            transforms.Resize((256, 256)),
            #transforms.ToTensor(),
        ])

    def forward(self, x):
        return self.base_model(x)


    def preprocess(self, img, xmin, ymin, xmax, ymax):
        print(type(img))
        img=np.array(img)
        print(xmin, ymin, xmax, ymax) 
        (h,w) = img.shape[:2]
        print(h,w)
        za = [xmin*w, ymin*h, xmax*w, ymax*h]
        print(za)
        im=Image.fromarray(img).crop(za)
        im = torch.from_numpy(np.array(self.transform(im)))
        im = torch.permute(im,(2, 0, 1)).unsqueeze(0)
        return(im)

    def predict(self,im):
        with torch.no_grad(): 
            im.to(self.device)
            raw_prob = self.forward(im.float()) 
            raw_prob = torch.sigmoid(raw_prob)
            print(raw_prob)
            raw_pred = torch.round(raw_prob)
            raw_prob = raw_prob.detach().numpy()[0] 
            raw_pred = raw_pred.detach().numpy()[0] 
            print(raw_prob)
            mlabel=str(int(raw_pred[0]))
            print(raw_pred)
            if (raw_pred==1):
                pred_lab = 'ПОДПИСАН'
            else:
                pred_lab = 'НЕ_ПОДПИСАН'
            np.round(raw_prob,decimals=3)
            rp = np.round(raw_prob,decimals=3)
            r = str(rp[0])
            print('r rounded list:',r)
            return {'label':  pred_lab, 'mlabel': mlabel, 'prob': r}
