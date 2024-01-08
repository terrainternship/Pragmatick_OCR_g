import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
#from trform import trf
from votemodules import image_align, vote, upload

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

vpath='weights/chpt35-col.pth'
vdim=3

DESCRIPTION = '''# <div align="center">Vote Inference Demo. by Yuri D.Kobyzev(c) v.0.0.1 </div>
<div align="center">
</div>

'''

if __name__ == '__main__':
    title = 'Vote Inference. by Kobyzev Yuri.D. Kobyzev'
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            #with gr.TabItem('Выравнивание документа'):
            #    image_align.Align()
            with gr.TabItem('Определение результата голосования'):
                vote.Vote(vpath,vdim,device)
            with gr.TabItem('Загрузки файлов'):
                upload.Upload()

#demo.queue().launch( debug=True, share=False)
demo.queue().launch(auth=("vote","vote"), debug=True, share=False, root_path="/vote")
