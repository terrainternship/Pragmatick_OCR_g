import gradio as gr
import os
import json
import numpy as np
import time
from PIL import Image
from .voteparser import vparser
from .votemodel import Resnext50ml, Resnet50s



class Vote:
    def __init__(self,device='cpu'):
        self.jpath='results'
        vmodel = Resnext50ml('weights/checkpoint-000035_sta0.pth',device, 4)
        smodel = Resnet50s('weights/vote_resnet50_sign.pt',device, 2)
        self.vp = vparser(vmodel, smodel)
        self.create_ui()
        pass

    def create_ui(self):

        def predict_vote(img,thr): 
            thr=thr/100
            print('THRESHOLD divided by 100:',thr)
            self.vp.process_data(img,thr)
            r=json.dumps(self.vp.pagevote)
            r = r.encode('utf-8').decode('unicode-escape')
            print(str(r))
            #self.vp.save_json_resut(jpath)
            #? annotate self.doc with bboxes
            return self.vp.doc[0],r


        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
            ** docTR amd resnet50 vote doc inference ** Kobyzev Yuri, Lysakov Yuri.
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

            gr.Markdown("## Image Examples")

            gr.Examples(
                examples=[["/vol/src/images/1-71-2-color.jpg",35]],
                inputs=[img_input,thr],
                outputs=[img_output,predict_json],
                fn=predict_vote,
                cache_examples=False,
           )
