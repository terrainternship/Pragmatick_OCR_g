import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
import cv2
import time 
from . import trf_utils


class Aline:
    def __init__(self):
        self.create_ui()
    
    def create_ui(self):

        def trns1(ref): 
            ref = trf_utils.fp_transf_bg(ref)
            return ref
        def trns2(ref): 
            ref = trf_utils.deskew_hough(ref)
            return ref
        def trns3(ref): 
            ref = trf_utils.deskew_fourier(ref)
            return ref



        with gr.Blocks() as demo:
            gr.Markdown("""
            ** Transform image example ** Kobyzev Yuri, Lysakov Yuri, Yulia D.
            """)

            with gr.Row():
              with gr.Column():
                  img_input = gr.Image()
                  aline1_btn = gr.Button("Выровнять Фото four_point (gray):")
                  aline2_btn = gr.Button("Выровнять Фото deskew hough (gray):")
                  aline3_btn = gr.Button("Выровнять Фото deskew fourier (gray):")

              with gr.Column():
                  img_output = gr.Image(image_mode='L')
                  clr_btn = gr.ClearButton([img_input,img_output],value = "Очистить")

            aline1_btn.click(trns1, [img_input], img_output)
            aline2_btn.click(trns2, [img_input], img_output)
            aline3_btn.click(trns3, [img_input], img_output)
            gr.Markdown("## Examples: four_points")
            gr.Examples( 
                    [["images/1-71-2-color.jpg"], ["images/1-71-2-vote.jpg"],["images/753_1.jpg"]],
                   [img_input],
                   [img_output],
                   fn=trns1,
                   cache_examples=False,
            )
            gr.Markdown("## Examples: descew hough")
            gr.Examples( 
                    [["images/1-71-2-color.jpg"], ["images/1-71-2-vote.jpg"],["images/753_1.jpg"]],
                   [img_input],
                   [img_output],
                   fn=trns2,
                   cache_examples=False,
            )
            gr.Markdown("## Examples: descew fourier")
            gr.Examples( 
                    [["images/1-71-2-color.jpg"], ["images/1-71-2-vote.jpg"],["images/753_1.jpg"]],
                   [img_input],
                   [img_output],
                   fn=trns3,
                   cache_examples=False,
            )

