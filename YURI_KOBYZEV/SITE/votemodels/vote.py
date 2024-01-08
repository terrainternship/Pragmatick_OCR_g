import gradio as gr
import os
import json
import numpy as np
import time
from PIL import Image
from .voteparser import vparser
from .padparser import padparser
from .votemodel import Resnext50ml, Resnet50s
import torch
import supervision as  sv
import cv2
from . import trf_utils

def binstr(ph):
    r=0
    for i in range(len(ph)):
        r = r+int(ph[i])*2**i
    return r   

def annotate(image_source: np.ndarray, boxes: torch.Tensor, phrases: list[str]) -> np.ndarray:
    h, w, _ = image_source.shape

    boxes = boxes * torch.Tensor([w, h, w, h])
    #xyxy = box_convert(boxes=boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
    xyxy = boxes.numpy()
    class_id=np.array([binstr(phrase) for phrase in phrases])
    detections = sv.Detections(xyxy=xyxy,class_id=class_id)
    labels = [
        f"{phrase}"
        for phrase
        in phrases
    ]
    box_annotator = sv.BoxAnnotator( )
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, 
            detections=detections, 
            labels=labels
            )
    return annotated_frame

def expand_greyscale_image_channels(grey_image):
    grey_image_arr = np.array(grey_image)
    grey_image_arr = np.expand_dims(grey_image_arr, -1)
    grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
    return grey_image_arr_3_channel


class Vote:
    def __init__(self,vpath,vdim,device):
        self.jpath='results'
        self.device=device
        self.vmodel = Resnext50ml(vpath,device,vdim)
        self.smodel = Resnet50s('weights/vote_bce_resnet50_sign.pt',device, 1)
        self.vp = vparser(self.vmodel, self.smodel)
        self.vpmodel='doctr'
        self.vpath=vpath
        self.spath='weights/vote_bce_resnet50_sign.pt'
        self.create_ui()
        pass


    def change_vmodel(self,vpath,dim): 
        self.vmodel = resnext50ml(vpath,self.device,dim)
#        self.vp = vparser(self.vmodel, self.smodel)
        self.vpath=vpath
        pass

    def change_vparser(self,vpmodel): 
        if vpmodel=='doctr': 
            self.vp = vparser(self.vmodel, self.smodel)
            self.vpmodel=vpmodel
        if vpmodel=='paddle': 
            self.vp = padparser(self.vmodel, self.smodel)
            self.vpmodel=vpmodel
        pass



    def predict_vote(self,img,thr,vmodelparams,vpmodel): 
        vpath,dim=vmodelparams.split(',')
        if self.vpath!=vpath:
            self.change_vmodel(vpath,int(dim))
        if self.vpmodel!=vpmodel:
            self.change_vparser(vpmodel)
        thr=thr/100
        self.vp.process_data(img,thr)
        r=json.dumps(self.vp.pagevote)
        r = r.encode('utf-8').decode('unicode-escape')
        bb=self.vp.annotate[0]['boxes']
        bb=torch.Tensor(np.array(bb))
        ph=self.vp.annotate[0]['phrases']
        annotated_frame = annotate(image_source=self.vp.doc[0], boxes=bb, phrases=ph)
        return annotated_frame,r

    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
            ** docTR amd resnet50 vote doc inference ** Kobyzev Yuri.
            """
            )

            with gr.Row():
                with gr.Column():
                    vmodelparams = gr.Dropdown(
                    label='Vote models', 
                    choices=['weights/chpt35-col.pth,3','weights/chpt10-bw.pth,3'],
                    value='weights/chpt35-col.pth,3',
                    )
                    vpmodel = gr.Dropdown(
                    label='Vote parser ocr', 
                    choices=['doctr','paddle'],
                    value='doctr',
                    )


                    thr = gr.Slider(1, 100, value=50, interactive=True, label="multilabel threshold", info="Choose between 1 and 99")
                    predict_btn = gr.Button("Предикт:")
                    align_btn = gr.Button("Выровнять {маркеры должны быть}")
                    img_input = gr.Image()
                    img_output = gr.Image()

                with gr.Column():
                    predict_json = gr.JSON()

            predict_btn.click(self.predict_vote, [img_input,thr,vmodelparams,vpmodel], [img_output,predict_json])
            align_btn.click(self.vp.alignimage, [img_input], [img_input])

            gr.Markdown("## Examples")
            gr.Examples( examples = [["images/voice0001-0.jpg"],["images/voice0001-1.jpg"],["images/voice0001-2.jpg"],["images/voice0002-0.jpg"],["images/voice0002-1.jpg"],["images/voice0002-2.jpg"],["images/voice0003-0.jpg"],["images/voice0003-1.jpg"],["images/voice0003-2.jpg"]],
                   inputs = img_input,
                   cache_examples=False,
            )
