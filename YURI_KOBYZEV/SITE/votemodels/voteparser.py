import os
import easyocr
import cv2
import imutils
import re
from pathlib import Path
os.environ['USE_TORCH'] = '1'
import torch
from doctr.io import DocumentFile
from transformations import Aligner
from doctr.models import ocr_predictor
import numpy as np
import json
from PIL import Image
import pytesseract

# bbox для номера голосования

bxmin, bymin, bxmax, bymax  = [0.40586210391716565, 0.974609375, 0.7997929921407185, 0.9892578125]

class  vparser:
    def __init__(self, vmodel, smodel):
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        self.vmodel=vmodel
        self.smodel=smodel
        self.aligner=Aligner()
        pass

    def preprocess_tesser(self,xmin, ymin, xmax, ymax):
        (h,w,c) = self.doc[0].shape
        h=h
        w=w
        za = [xmin*w, ymin*h, xmax*w, ymax*h]
        print("zalistnum:",za)
       
        im = Image.fromarray(self.doc[0]).crop(za)
        im = np.array(im)
        listnum = pytesseract.image_to_string(im,config ='--psm 6')
        print('listnum:',listnum,'len:',len(listnum))
        if len(listnum)==0:
            print("not found listnum")
            return 3
        listnum2=str(listnum[:2])
        listnum=str(listnum[0])
        if listnum2=='a:' or listnum2=='a?' or listnum2==';&' or listnum2=='-_':
            listnum=1
            print('LN2: tesser 1 a:==1 && ;&==1: ',listnum)
            return 1
        if listnum2=='a"':
            listnum=2
            print('LN2: tesser a"==2: ',listnum)
            return 2
        if listnum==':' or listnum=='i' or listnum=='I' or listnum==';' or listnum=='|':
            listnum=1
            print('LN1: tesser 1 :==1 ',listnum)
            return 1
        if listnum=='-':
            listnum=1
            print('LN1: tesser -==1 ',listnum)
            return 1
        if listnum=='a' or listnum=='\x0c':
            listnum=3
            print('LN1: tesser a==3: ',listnum)
            return 3 
        print("exit not found:",listnum)
        return listnum

    def preprocess_tesser_num(self,xmin, ymin, xmax, ymax):
        (h,w,c) = self.doc[0].shape
        za = [xmin*w, ymin*h, xmax*w, ymax*h]
        print("za:",za)
        im = Image.fromarray(self.doc[0]).crop(za)
        n = pytesseract.image_to_string(im,lang='eng',config ='--psm 6 --dpi 300 -c tessedit_char_whitelist=N0123456789-/').replace('\n\x0c', '')
        n = re.sub(r"\s+", "", n)
        print(type(n),n,len(n))
        n=n[1:]
        print('NUM: tesser: ',n)
        return(n)
    
    def find_listnum(self,pl,w,bb,ph): 
        listnum=pl['value']
        d = self.digit.search(listnum)
        print("listnum:",listnum)
        if (d == None or not (len(d.string)==1 and int(d.string)>0)): 
            xmin = w['geometry'][0][0]
            xmax = w['geometry'][1][0]
            ymin = w['geometry'][0][1]
            ymax = w['geometry'][1][1]
            #listnum = self.preprocess_tesser(xmin/3,ymin,2*xmin/3,ymax)
            listnum = self.preprocess_tesser(xmin/3,ymin,2*xmin/3,ymax)
            print("tesser_listnum=",listnum)
            #bb.append([xmin/3,ymin,2*xmin/3,ymax])
            bb.append([xmin/3,ymin,2*xmin/3,ymax])
            if listnum==1: 
                ph.append('100')
            else: 
                if listnum==2: 
                    ph.append('010')
                else: 
                    if listnum==3: 
                        ph.append('001') 
                    else: 
                        ph.append('111')
        return listnum,bb,ph

    def alignimage(self,doc_path): 
        print('align image:')
        if (isinstance(doc_path, str)):
            im = DocumentFile.from_images(doc_path)[0]
        else:
            im = doc_path
        return self.aligner.cropping_image_by_markers(im)
    
    def process_data(self,doc_path,thr):
        bb=[]
        lo=[]
        ph=[]
        print("process_data vthreshold:",thr)
        if (isinstance(doc_path, str)):
            self.doc = DocumentFile.from_images(doc_path)
            self.path=doc_path
        else:
            self.doc=[]
            self.doc.append(doc_path)
        self.data = self.model(self.doc).export()
        # '3881009-0397750-5915515/2-78'
        self.patterns=[re.compile('N.(\d{7}-\d{7}-\d{7}\/\d{1}-\d{2})'),re.compile('N.(\d{7}-\d{7}-\d{7}\d{1}-\d{2})'),re.compile('(\d{7}-\d{7}-\d{7}\d{1}-\d{2})'),re.compile('N(\d{7}-\d{7}-\d{7}\/\d{1}-\d{2})'),re.compile('N(\d{7}-\d{7}-\d{3}\d{7}\/\d{1}-\d{2})'),re.compile('(\d{7}-\d{7}-\d{3}\d{7}\d{1}-\d{2})')]
        self.digit=re.compile('(\d{1})')
        self.pagevote=[]
        self.annotate=[]
        sign_result=[]
        vote_result=[]
        bb=[]
        lo=[]
        ph=[]
        line=0
        vote_question_num=0
        for p in self.data['pages']:
            n=None
            listnum=None
            for b in p['blocks']:
                for l in b['lines']:
                    line+=1
                    for w in l['words']: 
                        print(w)            
                        if w['value']=='3A' or w['value'] == 'SA': 
                            vote_question_num+=1
                            xmin = w['geometry'][0][0]
                            xmax = w['geometry'][1][0]
                            ymin = w['geometry'][0][1]
                            ymax = w['geometry'][1][1]
                            deltay = (ymax-ymin)*0.5
                            bb.append([0.,ymin - deltay,1.,ymax +deltay])
                            timg =  self.vmodel.preprocess(self.doc[0], 0. , ymin - deltay, 1., ymax + deltay)
                            vresult = self.vmodel.predict(timg,thr)
                            print('vote_question_num:',vote_question_num,'result:',vresult)
                            #vr=vresult['label'].encode('utf-8').decode('unicode-escape')
                            ph.append(vresult['mlabel'])
                            vote_result.append(vresult)
                            break
                        if w['value'][1:]=='MO': 
                            xmin = w['geometry'][0][1]
                            xmax = w['geometry'][1][0]
                            ymin = w['geometry'][0][1]
                            ymax = w['geometry'][1][1]
                            deltah = ymax - ymin
                            timg =  self.smodel.preprocess(self.doc[0], 0.5, ymin - 4*deltah, 1, ymin + 2*deltah)
                            bb.append([0.5,ymin-4*deltah,1,ymin + 2*deltah])
                            sresult = self.smodel.predict(timg)
                            ph.append(sresult['mlabel'])
                            print(sresult)
                            sign_result.append(sresult)
                            break


                        
                        for i,p in enumerate(self.patterns):
                            nn = p.search(w['value'])
                            if nn and i==0:
                                print('match==',i)
                                n=nn.string[2:]
                                print("doc N:",n)
                                if line>3: 
                                    listnum,bb,ph = self.find_listnum(prevline,w,bb,ph)
                                break
                            if nn and i==1:
                                print('match==',i)
                                s1,s2,s3,s4 = nn.string[2:].split('-')
                                l3=len(s3)-1
                                n = s1+'-'+s2+'-'+s3[:l3]+'/'+s3[l3]+'-'+s4
                                print("doc N:",n)
                                if line>3: 
                                    listnum,bb,ph = self.find_listnum(prevline,w,bb,ph)
                                break
                            if nn and i==2: # insert
                                print('match==',i)
                                s1,s2,s3,s4 = nn.string.split('-')
                                l3=len(s3)-1
                                n = s1+'-'+s2+'-'+s3[:7]+'/'+s3[7]+'-'+s4
                                print("doc N:",n)
                                if line>3: 
                                    listnum,bb,ph = self.find_listnum(prevline,w,bb,ph)
                                break
                            if nn and i==3:
                                print('match==',i)
                                s1,s2,s3,s4 = nn.string[1:].split('-')
                                n = s1+'-'+s2+'-'+s3+'-'+s4
                                print("doc N:",n)
                                if line>3: 
                                    listnum,bb,ph = self.find_listnum(prevline,w,bb,ph)
                                break
                            if nn and i==4:
                                print('match==',i)
                                s1,s2,s3,s4 = nn.string[1:].split('-')
                                n = s1+'-'+s2+'-'+s3[3:]+'-'+s4
                                print("doc N:",n)
                                if line>3: 
                                    listnum,bb,ph = self.find_listnum(prevline,w,bb,ph)
                                break
                            if nn and i==5:
                                print('match==',i)
                                s1,s2,s3,s4 = nn.string.split('-')
                                n = s1+'-'+s2+'-'+s3[:7]+'/'+s3[7]+'-'+s4
                                print("doc N:",n)
                                if line>3: 
                                    listnum,bb,ph = self.find_listnum(prevline,w,bb,ph)
                                break
                        prevline=w
            if n == None:
                listnum = self.preprocess_tesser(bxmin/3,bymin,2*bxmin/3,bymax)
                bb.append([bxmin, bymin, bxmax, bymax])
                ph.append('111')
                bb.append([bxmin/3, bymin, 2*bxmin/3, bymax])
                ph.append('111')
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


