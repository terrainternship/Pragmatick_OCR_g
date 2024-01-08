import os
import easyocr
import cv2
import imutils
import re
from pathlib import Path
os.environ['USE_TORCH'] = '1'
import torch
from paddleocr import PaddleOCR
from transformations import Aligner
from doctr.models import ocr_predictor
import numpy as np
import json
from PIL import Image
import pytesseract

class  padparser:
    def __init__(self, vmodel, smodel):
        self.model = PaddleOCR(use_angle_cls=False,use_gpu=False,lang='en',det_db_thresh=0.05, det_db_box_thresh=0.1)
        self.vmodel=vmodel
        self.smodel=smodel
        self.aligner=Aligner()
        pass

    def alignimage(self,doc_path): 
        print('align image:')
        if (isinstance(doc_path, str)):
            im = cv2.imread(doc_path)
        else:
            im = doc_path
        return self.aligner.cropping_image_by_markers(im)
    
    def process_data(self,doc_path,thr):
        bb=[]
        lo=[]
        ph=[]
        print("process_data vthreshold:",thr)
        if (isinstance(doc_path, str)):
            self.doc = [cv2.imread(doc_path)]
            self.path=doc_path
        else:
            self.doc = [doc_path]
        self.data = self.model.ocr(self.doc[0])
        (h,w,_)=self.doc[0].shape
        # 'N.3881009-0397750-5915515/2-78'
        # 'N6224536-6062231-7990235/2-78'
        self.p=re.compile('N(\d{7}-\d{7}-\d{7}\/\d{1}-\d{2})')
        self.pp=re.compile('N.(\d{7}-\d{7}-\d{7}\/\d{1}-\d{2})')
        self.pagevote=[]
        self.annotate=[]
        sign_result=[]
        vote_result=[]
        bb=[]
        lo=[]
        ph=[]
        line=0
        listnum=None
        n = None

        for r in self.data[0]:
            # RRR===> [[[44.0, 13.0], [380.0, 13.0], [380.0, 26.0], [44.0, 26.0]], ('6.2 Iopacxa arossx axr-177ks.M.croocrbo 35400 py6.00 ko', 0.8241300582885742)]
            line+=1
            #print("line:",line,"box:",r[0], " value",r[1][0], " prob:",r[1][1])
            print("r0===",r)
            xmin=r[0][0][0]/w
            ymin=r[0][0][1]/h
            xmax=r[0][2][0]/w
            ymax=r[0][2][1]/h
            value=r[1][0]
            if value=='3A': 
                deltay = (ymax-ymin)*1.0
                bb.append([0.,ymin - deltay,1.,ymax +deltay])
                timg =  self.vmodel.preprocess(self.doc[0], 0. , ymin - deltay, 1., ymax + deltay)
                vresult = self.vmodel.predict(timg,thr)
                ph.append(vresult['mlabel'])
                vote_result.append(vresult)
            if value[1:]=='NO': 
                deltah = ymax - ymin
                timg =  self.smodel.preprocess(self.doc[0], 0.5, ymin - 4*deltah, 1, ymin + 2*deltah)
                bb.append([0.5,ymin-4*deltah,1,ymin + 2*deltah])
                sresult = self.smodel.predict(timg)
                ph.append(sresult['mlabel'])
                print(sresult)
                sign_result.append(sresult)
                
            nn = self.p.search(value)
            if nn:
                print(nn.start()+2,nn.end())
                n=nn.string[nn.start()+1:nn.end()]
                print("doc N1:",n)
                if line>3: 
                    listnum = prevline
                    bb.append(prevbox)
                    ph.append('111')
            else: 
                nn = self.pp.search(value)
                if nn:
                    #n=nn.string[2:]
                    print(nn.start()+2,nn.end())
                    n=nn.string[nn.start()+2:nn.end()]
                    print("doc N2:",n)
                    if line>3: 
                        listnum = prevline
                        bb.append(prevbox)
                        ph.append('111')

            prevline=value
            prevbox=[xmin,ymin,xmax,ymax]
        bbb=bb[:len(vote_result)]    
        self.pagevote.append({'list_N': listnum,'NUM': n,'vote_result': vote_result,'box': bbb,'sign_resut': sign_result}) # for each doc
        print(self.pagevote)
        self.annotate.append({'boxes': bb,'phrases': ph}) # for each doc
        print(self.annotate)
            

    def save_json_resut(self,jpath):
        # self.pagevote ---> json
        r=json.dumps(self.pagevote)
        r = r.encode('utf-8').decode('unicode-escape')
        fname=Path(self.path).stem
        s = self.pagevote[0]["NUM"]
        l ='ln-'
        listnum = self.pagevote[0]["list_N"]
        print(self.pagevote[0])
        if s !=None: 
            s ='num-'+s.replace('/','-') 
        else:
            s='num-'
        print('check listnum:',listnum)
        if listnum!=None:
            l = 'ln-'+str(listnum)
        fname = fname
        fname=fname+'R'+s+'-'+l+'.json'
        fname=os.path.join(jpath,fname)
        with open(fname,'w') as jtxt:
            jtxt.write(r)


