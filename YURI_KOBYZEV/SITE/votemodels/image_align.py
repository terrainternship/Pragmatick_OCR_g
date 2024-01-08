import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
import cv2
import time 
from . import trf_utils
from transformations import Aligner

def trns1(ref): 
    ref = trf_utils.fp_transf_bg(ref)
    return ref

def trns2(ref): 
    ref = trf_utils.deskew_hough(ref)
    return ref

def trns3(ref,minwidth,step,thr1,thr2,maxdelta): 
    thr1=thr1/100
    thr2=thr2/100
    print("minwidth step thr1,thr2,maxdelta",minwidth,step,thr1,thr2,maxdelta)
    aligner=Aligner(minwidth,step,thr1,thr2,maxdelta)
    ref = aligner.cropping_image_by_markers(ref)
    return ref



class Align:
    def __init__(self):
        self.create_ui()
        pass

    def create_ui(self):

        with gr.Blocks() as demo:
            gr.Markdown("""
            ** Transform image example ** Kobyzev Yuri, Lysakov Yuri, Yulia D.
            """)

            with gr.Row():
              with gr.Column():
                  img_input = gr.Image()

              with gr.Column():
                  img_output = gr.Image(image_mode='L')
                  clr_btn = gr.ClearButton([img_input,img_output],value = "Очистить")
            with gr.Row():
              with gr.Column():
                  aline1_btn = gr.Button("Выровнять Фото four_point (gray):")
                  aline2_btn = gr.Button("Выровнять Фото deskew hough (gray):")
                  minwidth = gr.Slider(600, 900, value=700, interactive=True, label="MIN_WIDTH", info="Стартовая ширина масштабирования изображения при поиске маркеров 600 - 900 (700)")
                  step = gr.Slider(1, 20, value=10, interactive=True, label="STEP", info="Приращение ширины для следующей итерации 1 - 20 (10)")
                  thr1 = gr.Slider(0, 100, value=80, interactive=True, label="THRESHOLD_1", info=" Максимальная разница между соответствующими координатами центров маркеров 0. - 1. (0.8)")
                  thr2 = gr.Slider(0, 100, value=85, interactive=True, label="THRESHOLD_2", info=" Максимальная разница между соответствующими координатами центров маркеров 0. - 1. (0.85)")
                  maxdelta = gr.Slider(0, 10, value=5, interactive=True, label="MAX_DELTA", info=" Максдэлта 0 - 10 (5)")
                  aline3_btn = gr.Button("Выровнять Фото markers four_point (gray):")

            aline1_btn.click(trns1, [img_input], img_output)
            aline2_btn.click(trns2, [img_input], img_output)
            aline3_btn.click(trns3, [img_input,minwidth,step,thr1,thr2,maxdelta], img_output)
            gr.Markdown("## Examples: four_points")
            gr.Examples( examples = [["images/voice0001-0.jpg"],["images/voice0001-1.jpg"],["images/voice0001-2.jpg"],["images/voice0002-0.jpg"],["images/voice0002-1.jpg"],["images/voice0002-2.jpg"],["images/voice0003-0.jpg"],["images/voice0003-1.jpg"],["images/voice0003-2.jpg"]],
                   inputs = img_input,
                   cache_examples=False,
            )
