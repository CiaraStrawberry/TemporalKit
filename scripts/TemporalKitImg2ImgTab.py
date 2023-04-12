import gradio as gr
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import modules.scripts as scripts
import gradio as gr
from modules import processing, images, shared, sd_samplers, devices
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.images as images
import os
import glob
lastimage = None

def save_image(image, filename, directory):
    os.makedirs(directory, exist_ok=True)

    # create the full file path by joining the directory and filename
    file_path = os.path.join(directory, filename)

    # save the image to the specified file path
    image.save(file_path)

def on_button_click():
    global lastimage
    extension_path = os.path.abspath(__file__)
    extension_dir =  os.path.dirname(os.path.dirname(extension_path))
    extension_folder = os.path.join(extension_dir,"squares")
    
    # save the image to the parent directory with a new filename
    save_image(lastimage, 'last.png', extension_folder)

        
class Script(scripts.Script):
    global button
    def title(self):
        return "TemporalKit"

    def show(self, is_img2img):
        if is_img2img:
            return True
        return True


    def ui(self, is_img2img): 
        global lastimage
        savebutton = gr.Button("save", label="Save")
        savebutton.click(
            fn=on_button_click,
        )
        movebutton = gr.Button("move", label="Move")
        movebutton.click( 
            fn=None,
            _js="switch_to_temporal_kit",
        )  


    def run(self, p):
        global lastimage
        processed = processing.process_images(p)
        lastimage = processed.images[0]
        #registerbuttons(button)
        return processed

