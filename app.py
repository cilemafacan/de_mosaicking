import gradio as gr
import numpy as np
from PIL import Image
from demosaicking import mosaicking, de_mosaicking, check_input, padding
from metrics import psnr, mae, mse

inp_image = Image.open("images/medium.png")
bayer = np.array([[[0.,1.], [0.,0.]], 
                    [[1.,0.], [0.,1.]], 
                    [[0.,0.], [1.,0.]]]).transpose(1, 2, 0).astype(np.float32)

def apply_mosaicking(input_image, bayer_pattern):
    check_input(input_image)
    np_input_image = np.array(input_image, dtype=np.float32)
    np_input_image = padding(np_input_image)
    mosaic_image = mosaicking(np_input_image, bayer_pattern)
    return Image.fromarray(mosaic_image.astype(np.uint8))

def apply_de_mosaicking(mosaic_image):
    demosaic_image = de_mosaicking(mosaic_image)
    return demosaic_image

with gr.Blocks() as demo:
    with gr.Row():
        gr.Label("Demosaicing", show_label=False)
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(inp_image, label="Original Image", type="pil", interactive=False)
        with gr.Column(scale=2):
            bayer_pattern = gr.Image(bayer, label="Bayer Pattern", type="numpy", scale=1000, interactive=False)
            mosaic_image = gr.Image(label="Mosaic Image", type="pil", interactive=False)
        with gr.Column(scale=2):
            demosaic_image = gr.Image(label="De-Mosaic Image", type="pil", interactive=False)
        with gr.Column(scale=1):
            button_mosaic = gr.Button(value="Apply Mosaicing", interactive=True)
            button_demosaic = gr.Button(value="Apply De-Mosaicing", interactive=True)
            info_box = gr.Textbox(label="Info", interactive=False)
            psnr_box = gr.Textbox(label="PSNR", interactive=False)
            mse_box = gr.Textbox(label="MSE", interactive=False)
            mae_box = gr.Textbox(label="MAE", interactive=False)

    button_mosaic.click(apply_mosaicking, inputs=[input_image, bayer_pattern], outputs=[mosaic_image])
    button_demosaic.click(apply_de_mosaicking, inputs=[mosaic_image], outputs=[demosaic_image])

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)