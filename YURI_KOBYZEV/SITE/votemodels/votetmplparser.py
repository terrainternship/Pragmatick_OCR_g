import os
from pathlib import Path
import numpy as np
import json



class  votetmplarser:
    def __init__(self,tmpl_num, vmodel, smodel, ocrmodel):
        self.model = ocrmodel
        self.vmodel = vmodel
        self.smodel = smodel
        self.tmpl_num=tmpl_num
        with open(str(tmpl_num)+'.json','r') as tmpl:
            self.tmpl=json.loads(tmpl.read())
        pass
    
    def process_data(self,image,thr):
        bb=[]
        lo=[]
        ph=[]
        self.doc=[]
        self.doc.append(doc_path)
        self.data = self.model(self.doc).export()
        print("PAGES:",self.data['pages'])
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
                       # print(w)            
                        if w['value']=='3A': 
                            vote_question_num+=1
                            xmin = w['geometry'][0][0]
                            ymin = w['geometry'][0][1]
                            ymax = w['geometry'][1][1]
                            bb.append([3*xmin/4,ymin,1.-3*xmin/4,ymax])
                            timg =  self.vmodel.preprocess(self.doc[0], 3*xmin/4 , ymin, 1-3*xmin/4, ymax)
                            #print("vthreshold:",thr)
                            vresult = self.vmodel.predict(timg,thr)
                            print('vote_question_num:',vote_question_num,'result:',vresult)
                            ph.append(vresult['mlabel'])
                            vote_result.append(vresult)
                        if w['value'].startswith('N'):
                            lenval=len(w['value'])
                            n=w['value'][2:lenval]
                            if lenval>=28:
                                print("doc N:",n)
                                if line>3:
                                    listnum=prevline['value']
                                    print("pagelist N:",listnum)
                        #if w['value']=='@MO': 
                        if w['value']=='nonvcb': 
                            xmin = w['geometry'][0][1]
                            xmax = w['geometry'][1][0]
                            ymin = w['geometry'][0][1]
                            ymax = w['geometry'][1][1]
                            deltah = ymax - ymin
                            timg =  self.smodel.preprocess(self.doc[0], 0.5, ymin - 3*deltah, 1, ymin - deltah)
                            #bb.append([0.5,ymin-3*deltah,1,ymin - deltah])
                            sresult = self.smodel.predict(timg)
                            #ph.append(sresult['label'])
                            print(sresult)
                            sign_result.append(sresult)
                        if w['value']=='CO6CTBEHHHK':
                            xmin = w['geometry'][0][1]
                            xmax = w['geometry'][1][0]
                            ymin = w['geometry'][0][1]
                            ymax = w['geometry'][1][1]
                            deltah = ymax - ymin
                            timg =  self.smodel.preprocess(self.doc[0], 0.5, ymin - 3*deltah, 1, ymin - deltah)
                            #bb.append([0.5,ymin-3*deltah,1,ymin - deltah])
                            sresult = self.smodel.predict(timg)
                            #lo.append(annoresult['logit'])
                            #ph.append(sresult['label'])
                            print(sresult)
                            sign_result.append(sresult)

                        prevline=w
            self.pagevote.append({'listnum': [listnum,n],'vote_result': vote_result,'sign_resut': sign_result}) # for each doc
            self.annotate.append({'boxes': bb,'phrases': ph}) # for each doc
            



    def save_json_resut(self,jpath):
        # self.pagevote ---> json
        r=json.dumps(self.pagevote)
        r = r.encode('utf-8').decode('unicode-escape')
        fname=Path(self.path).stem+'.json'
        fname=os.path.join(jpath,fname)
        with open(fname,'w') as jtxt:
            jtxt.write(r)


