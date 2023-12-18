import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
#from trform import trf
from votemodules import vote,aline


device='cuda'
device='cpu'

DESCRIPTION = '''# <div align="center">Vote Inference Demo. by Yuri D.Kobyzev(c) v.0.0.1 </div>
<div align="center">
</div>

'''

if __name__ == '__main__':
    title = 'Vote Inference. by Kobyzev Yuri.D. Kobyzev'
    with gr.Blocks(analytics_enabled=False, title=title, theme=gr.themes.Monochrome()) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('Выравнивание документа'):
                aline.Aline()
            with gr.TabItem('Определение результата голосования'):
                vote.Vote(device)

#demo.queue().launch(share=False,debug=True)
demo.queue().launch(auth=("vote","vote"), debug=True, share=False, root_path="/vote")
