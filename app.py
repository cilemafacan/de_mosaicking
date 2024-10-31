import gradio as gr
import numpy as np
from PIL import Image
from demosaicking import mosaicking, de_mosaicking, check_input, padding
from metrics import psnr, mae, mse

inp_image = Image.open("images/medium.png")

def apply_mosaicking(input_image):
    bayer = np.array([[[0.,1.], [0.,0.]], 
                     [[1.,0.], [0.,1.]], 
                     [[0.,0.], [1.,0.]]]).transpose(1, 2, 0).astype(np.float32)
    np_input_image = input_image.astype(np.float32)
    np_input_image = padding(np_input_image)  
    mos_image = mosaicking(np_input_image, bayer)
    mos_image = mos_image.astype(np.uint8)
    return mos_image

def apply_de_mosaicking(mosaic_image):
    mosaic_image = mosaic_image.astype(np.float32)
    demosaic_image = de_mosaicking(mosaic_image)
    demosaic_image = demosaic_image.astype(np.uint8)
    return demosaic_image

def calculate_metrics(input_image, demosaic_image):
    input_image = input_image.astype(np.float32)
    demosaic_image = demosaic_image.astype(np.float32)
    psnr_value = psnr(input_image, demosaic_image)
    mse_value = mse(input_image, demosaic_image)
    mae_value = mae(input_image, demosaic_image)
    return psnr_value, mse_value, mae_value

with gr.Blocks() as demo:
    with gr.Row():
        gr.Label("Demosaicing", show_label=False)
    with gr.Row():
        with gr.Column(scale=4):
            input_image = gr.Image(inp_image, label="Original Image", format="png", interactive=False, height=800)
        with gr.Column(scale=4):
            mosaic_image = gr.Image(label="Mosaic Image", format="png",interactive=False, height=800)
        with gr.Column(scale=4):
            demosaic_image = gr.Image(label="De-Mosaic Image", format="png", interactive=False, height=800)
        with gr.Column(scale=1):
            button_mosaic = gr.Button(value="Apply Mosaicing", interactive=True)
            button_demosaic = gr.Button(value="Apply De-Mosaicing", interactive=True)
            button_metrics = gr.Button(value="Calculate Metrics", interactive=True)
            psnr_box = gr.Textbox(label="PSNR", interactive=False)
            mse_box = gr.Textbox(label="MSE", interactive=False)
            mae_box = gr.Textbox(label="MAE", interactive=False)

    button_mosaic.click(apply_mosaicking, inputs=[input_image], outputs=[mosaic_image])
    button_demosaic.click(apply_de_mosaicking, inputs=[mosaic_image], outputs=[demosaic_image])
    button_metrics.click(calculate_metrics, inputs=[input_image, demosaic_image], outputs=[psnr_box, mse_box, mae_box])


demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
