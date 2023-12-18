import gradio as gr
import os
import json
import numpy as np
import time
from PIL import Image
from .voteparser import vparser
from .votemodel import Resnext50ml, Resnet50s
import torch
import supervision as  sv
import cv2


def annotate(image_source: np.ndarray, boxes: torch.Tensor, phrases: list[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    #print(boxes)

    boxes = boxes * torch.Tensor([w, h, w, h])
    #xyxy = box_convert(boxes=boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
    xyxy = boxes.numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase}"
        for phrase
        in phrases
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame



class Vote:
    def __init__(self,device='cpu'):
        self.jpath='results'
       # vmodel = Resnext50ml('weights/chpt35-col.pth',device, 3)
       # vmodel = Resnext50ml('weights/chpt10-bw.pth',device, 3)
        vmodel = Resnext50ml('weights/checkpoint-000035_sta0.pth',device, 4)
        smodel = Resnet50s('weights/vote_resnet50_sign.pt',device, 2)
        self.vp = vparser(vmodel, smodel)
        self.create_ui()
        pass

    def create_ui(self):

        def predict_vote(img,thr): 
            thr=thr/100
            self.vp.process_data(img,thr)
            r=json.dumps(self.vp.pagevote)
            r = r.encode('utf-8').decode('unicode-escape')
            #print(str(r))
            #self.vp.save_json_resut(jpath)
            #annotate self.doc with bboxes

            bb=self.vp.annotate[0]['boxes']
            bb=torch.Tensor(np.array(bb))
            ph=self.vp.annotate[0]['phrases']
            #print(bb)
            #print(ph)

            annotated_frame = annotate(image_source=self.vp.doc[0], boxes=bb, phrases=ph)
  #return (annotated_frame)
            return annotated_frame,r


        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
            ** docTR amd resnet50 vote doc inference ** Kobyzev Yuri.
            """
            )

            with gr.Row():
                with gr.Column():
                    thr = gr.Slider(1, 90, value=35, interactive=True, label="multilabel threshold", info="Choose between 1 and 99")
                    img_input = gr.Image()
                    img_output = gr.Image()
                    predict_btn = gr.Button("Предикт:")

                with gr.Column():
                    predict_json = gr.JSON()
            clr_btn = gr.ClearButton([img_input, img_output, predict_json], value="Очистить")

            predict_btn.click(predict_vote, [img_input,thr], [img_output,predict_json])

            gr.Markdown("## Examples")
            gr.Examples( 
                    [["images/1-71-2-color.jpg", "0.5"], ["images/1-71-2-vote.jpg", "0.5"],["images/753_1.jpg",0.5]],
                   [img_input,predict_json],
                   [img_output,predict_json],
                   fn=predict_vote,
                   cache_examples=False,
            )
