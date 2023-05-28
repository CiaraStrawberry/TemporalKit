from __future__ import annotations
import math
import random
import sys
from argparse import ArgumentParser
from collections import namedtuple, deque
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import os
import shutil
import time
import stat
import gradio as gr
import modules.extras
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.ui import create_toprow, create_sampler_and_steps_selection
import json
from modules.sd_samplers import samplers, samplers_for_img2img
import re
import modules.images as images
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules import ui_extra_networks, devices, shared, scripts, script_callbacks, sd_hijack_unet, sd_hijack_utils
from modules.shared import opts, cmd_opts, OptionInfo
from pathlib import Path
from typing import List, Tuple
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngImageFile, PngInfo
from datetime import datetime
from modules.generation_parameters_copypaste import quote
from copy import deepcopy
import platform
import modules.generation_parameters_copypaste as parameters_copypaste
import scripts.Berry_Method as General_SD
import glob
import base64
import io
import scripts.Ebsynth_Processing as ebsynth
import scripts.berry_utility as sd_utility


diffuseimg = None
SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])
lastmadefilename = ""
def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def preprocess_video(video,fps,batch_size,per_side,resolution,batch_run,max_frames,output_path,border_frames,ebsynth_mode,split_video,split_based_on_cuts):
    input_folder_loc = os.path.join(output_path, "input")
    output_folder_loc = os.path.join(output_path, "output")
    if not os.path.exists(input_folder_loc):
        os.makedirs(input_folder_loc)
    if not os.path.exists(output_folder_loc):
        os.makedirs(output_folder_loc)
    
    max_keys = max_frames
    if max_keys < 0:
        max_keys = 100000
    max_frames = (max_frames * (batch_size))
    if max_frames < 1:
        max_frames = 100000
    # would use mathf.inf in c#, dunno what that is in python
    # potential bug later, low priority
    if ebsynth_mode == True:
        if split_video == False:
            border_frames = 0
        if batch_run == False:
            max_frames = per_side * per_side * (batch_size + 1)

        
    if split_video == True:

        #max frames only applies in batch mode
        #otherwise it limmits the number of *total frames*
        border_frames = border_frames * batch_size

        max_frames = (20 * batch_size) - border_frames
        max_total_frames = int((max_keys / 20) * max_frames)
        existing_frames = [] 
           
        if split_based_on_cuts == True:
            existing_frames = sd_utility.split_video_into_numpy_arrays(video,fps,False)
        else:
            data = General_SD.convert_video_to_bytes(video)
            existing_frames = [sd_utility.extract_frames_movpie(data, fps,max_frames=max_total_frames,perform_interpolation=False)]
    

        split_video_paths,transition_data = General_SD.split_videos_into_smaller_videos(max_keys,existing_frames,fps,max_frames,output_path,border_frames,split_based_on_cuts)
        for index,individual_video in enumerate(split_video_paths):
            
            generated_textures = General_SD.generate_squares_to_folder(individual_video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side,max_frames=None, output_folder=os.path.dirname(individual_video),border=0, ebsynth_mode=ebsynth_mode,max_frames_to_save=max_frames)
            input_location = os.path.join(os.path.dirname(os.path.dirname(individual_video)),"input")
            for tex_index,texture in enumerate(generated_textures):
                individual_file_name = os.path.join(input_location,f"{index}and{tex_index}.png")
                General_SD.save_square_texture(texture,individual_file_name)
        transitiondatapath = os.path.join(output_path,"transition_data.txt")
        with open(transitiondatapath, "w") as f:
            f.write(str(transition_data) + "\n")
            f.write(str(border_frames) + "\n")
        main_video_path = os.path.join(output_path,"main_video.mp4")
        sd_utility.copy_video(video,main_video_path)
        return main_video_path
    
    new_video_loc = os.path.join(output_path, f"input_video.mp4")
    shutil.copyfile(video,new_video_loc)
    if ebsynth_mode == True:
        border = 0
        
        image = General_SD.generate_squares_to_folder(video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side,max_frames=max_frames, output_folder=output_path,border=border_frames, ebsynth_mode=True,max_frames_to_save=max_frames)
        return image[0]

    if batch_run == False:
        image = General_SD.generate_square_from_video(video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side )
        processed = numpy_array_to_temp_url(image)
    else:
        image = General_SD.generate_squares_to_folder(video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side,max_frames=max_frames, output_folder=output_path,border=border_frames,ebsynth_mode=False,max_frames_to_save=max_frames)
        processed = image[0]
    return processed



def apply_image_to_video(image,video,fps,per_side,output_resolution,batch_size):
    return General_SD.process_video_single(video_path=video,fps=fps,per_side=per_side,batch_size=batch_size,fillindenoise=0,edgedenoise=0,_smol_resolution=output_resolution,square_texture=image)

def apply_image_to_vide_batch(input_folder,video,fps,per_side,output_resolution,batch_size,max_frames,border_frames):
    input_images_folder = os.path.join (input_folder,"output")
    images = read_images_folder(input_images_folder)
    print(len(images))
    return General_SD.process_video_batch(video_path_old=video,fps=fps,per_side=per_side,batch_size=batch_size,fillindenoise=0,edgedenoise=0,_smol_resolution=output_resolution,square_textures=images,max_frames=max_frames,output_folder=input_folder,border=border_frames)

def post_process_ebsynth(input_folder,video,fps,per_side,output_resolution,batch_size,max_frames,border_frames):
    input_images_folder = os.path.join (input_folder,"output")
    images = read_images_folder(input_images_folder)
    print(len(images))
    split_mode = os.path.join(input_folder, "keys")
    if os.path.exists(split_mode):
        return ebsynth.sort_into_folders(video_path=video,fps=fps,per_side=per_side,batch_size=batch_size,_smol_resolution=output_resolution,square_textures=images,max_frames=max_frames,output_folder=input_folder,border=border_frames)
    else:
        img_folder = os.path.join(input_folder, "output")
        # define a regular expression pattern to match directory names with one or more digits
        pattern = r'^\d+$'

        # get a list of all directories in the specified path
        all_dirs = os.listdir(input_folder)

        # use a list comprehension to filter the directories based on the pattern
        numeric_dirs = sorted([d for d in all_dirs if re.match(pattern, d)], key=lambda x: int(x))
        max_frames = max_frames + border_frames
        for d in numeric_dirs:
            # create a list to store the filenames of the images that match the directory name
            img_names = []
            folder_video = os.path.join(input_folder, d, "input_video.mp4")
            # loop through each image file in the image folder
            for img_file in os.listdir(img_folder):
                # check if the image filename starts with the directory name followed by the word "and" and a sequence of one or more digits, then ends with '.png'
                if re.match(f"^{d}and\d+.*\.png$", img_file):
                    img_names.append(img_file)
            print(f"post processing = {os.path.dirname(folder_video)}")
            square_textures = []
            # loop through each image file name
            for img_name in sorted(img_names, key=lambda x: int(re.search(r'and(\d+)', x).group(1))):
                img = Image.open(os.path.join(input_images_folder, img_name))
                # Convert image to NumPy array and append to images list
                print(f"saving {os.path.join(input_images_folder, img_name)}")
                square_textures.append(np.array(img))

            ebsynth.sort_into_folders(video_path=folder_video, fps=fps, per_side=per_side, batch_size=batch_size,
                                    _smol_resolution=output_resolution, square_textures=square_textures,
                                    max_frames=max_frames, output_folder=os.path.dirname(folder_video),
                                    border=border_frames)

def recombine_ebsynth(input_folder,fps,border_frames,batch):
    if os.path.exists(os.path.join(input_folder, "keys")):
        return ebsynth.crossfade_folder_of_folders(input_folder,fps=fps,return_generated_video_path=True)
    else:
        generated_videos = []
        pattern = r'^\d+$'

        # get a list of all directories in the specified path
        all_dirs = os.listdir(input_folder)

        # use a list comprehension to filter the directories based on the pattern
        numeric_dirs = sorted([d for d in all_dirs if re.match(pattern, d)], key=lambda x: int(x))

        for d in numeric_dirs:
            folder_loc = os.path.join(input_folder,d)
            # loop through each image file in the image folder
            new_video =  ebsynth.crossfade_folder_of_folders(folder_loc,fps=fps)
            #print(f"generated new video at location {new_video}")
            generated_videos.append(new_video)
        
        overlap_data_path = os.path.join(input_folder,"transition_data.txt")
        with open(overlap_data_path, "r") as f:
            merge = str(f.readline().strip())

        overlap_indicies = []
        int_list = eval(merge)
        for num in int_list:
            overlap_indicies.append(int(num))



        output_video = sd_utility.crossfade_videos(video_paths=generated_videos,fps=fps,overlap_indexes= overlap_indicies,num_overlap_frames= border_frames,output_path=os.path.join(input_folder,"output.mp4"))
        return output_video
    return None


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def read_images_folder(folder_path):
    images = []
    filenames = os.listdir(folder_path)
    
    # Sort filenames based on the order of the numbers in their names
    filenames.sort(key=natural_keys)

    for filename in filenames:
        # Check if file is an image (assumes only image files are in the folder)
        if (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')) and (not re.search(r'-\d', filename)):
            if filename.startswith('input'):
                # Open image using Pillow library
            
                img = Image.open(os.path.join(folder_path, filename))
                # Convert image to NumPy array and append to images list
                images.append(np.array(img))
    return images



def numpy_array_to_data_uri(img_array):
    # convert the array to an image using PIL
    img = Image.fromarray(img_array)

    # create a BytesIO object to hold the image data
    buffer = io.BytesIO()

    # save the image to the BytesIO object as PNG
    img.save(buffer, format='PNG')

    # get the PNG data from the BytesIO object
    png_data = buffer.getvalue()

    # convert the PNG data to base64-encoded string
    base64_str = base64.b64encode(png_data).decode()

    # combine the base64-encoded string with the image format prefix
    data_uri = 'data:image/png;base64,' + base64_str
    
    return data_uri

def numpy_array_to_temp_url(img_array):
    # create a filename for the temporary file
    filename = 'generatedsquare.png'
    extension_path = os.path.abspath(__file__)
    extension_dir =  os.path.dirname(os.path.dirname(extension_path))
    extension_folder = os.path.join(extension_dir,"squares")  
    if not os.path.exists(extension_folder):
        os.makedirs(extension_folder)  
    # create a path for the temporary file
    file_path = os.path.join(extension_folder, filename)

    # convert the array to an image using PIL
    img = Image.fromarray(img_array)

    # save the image to the temporary file as PNG
    img.save(file_path, format='PNG')

    # create a URL for the temporary file
    #url = 'file://' + file_path
    
    return file_path

def display_interface(interface):
    return interface.display()


def get_most_recent_file(provided_directory):
    if not os.path.exists(provided_directory) or not os.path.isdir(provided_directory):
        raise ValueError("Invalid directory provided")

    # Get all files in the provided directory
    files = glob.glob(os.path.join(provided_directory, '*'))
    
    if not files:
        return None

    # Sort the files based on modification time and get the most recent one
    most_recent_file = max(files, key=os.path.getmtime)

    return most_recent_file

def update_image():
    global diffuseimg 
    extension_path = os.path.abspath(__file__)
    # get the directory name of the extension
    extension_dir =  os.path.dirname(os.path.dirname(extension_path))
    extension_folder = os.path.join(extension_dir,"squares")
    most_recent_image = get_most_recent_file(extension_folder)
    print(most_recent_image)
    pilImage = Image.open(most_recent_image)
    print("running")
    return most_recent_image

def update_settings():
    extension_path = os.path.abspath(__file__)
    extension_dir =  os.path.dirname(os.path.dirname(extension_path))
    tempfile = os.path.join(extension_dir,"temp_file.txt")
    with open(tempfile, "r") as f:
        fps = int(f.readline().strip())
        sides = int(f.readline().strip())
        batch_size = int(f.readline().strip())
        video_path = f.readline().strip()
    return fps,sides,batch_size,video_path

def update_settings_from_file(folderpath):
    read_path = os.path.join(folderpath,"batch_settings.txt")
    border = None
    print (f"batch settings exists = {os.path.exists(read_path)}")
    if os.path.exists(read_path) == False:
        read_path = os.path.join(folderpath,"0","batch_settings.txt")
        video_path = os.path.join(folderpath,"main_video.mp4")
        transition_data_path = os.path.join(folderpath,"transition_data.txt")
        if os.path.exists(transition_data_path):
            with open(transition_data_path, "r") as b:
                merge = str(b.readline().strip())
                border = int(b.readline().strip())
    print (f"reading path at {read_path}")
    with open(read_path, "r") as f:
        fps = int(f.readline().strip())
        sides = int(f.readline().strip())
        batch_size = int(f.readline().strip())
        video_path = f.readline().strip()
        max_frames = int(f.readline().strip())
        if border == None:
            border = int(f.readline().strip())


    return fps,sides,batch_size,video_path,max_frames,border


def save_settings(fps,sides,batch_size,video):
    extension_path = os.path.abspath(__file__)
    extension_dir =  os.path.dirname(os.path.dirname(extension_path))
    tempfile = os.path.join(extension_dir,"temp_file.txt")
    with open(tempfile, "w") as f:
        f.write(str(fps) + "\n")
        f.write(str(sides) + "\n")
        f.write(str(batch_size) + "\n")
        f.write(str(video) + "\n")

def create_video_Processing_Tab():
    with gr.Column(visible=True, elem_id="Temporal_Kit") as main_panel:
        dummy_component = gr.Label(visible=False)
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_TemporalKit", label="Input"):
                        with gr.Row():
                            with gr.Column():
                                video = gr.Video(label="Input Video", elem_id="input_video",type="filepath")
                                with gr.Row():
                                    sides = gr.Number(value=2,label="Sides", precision=0, interactive=True)
                                    resolution = gr.Number(value=1024,label="Height Resolution", precision=1, interactive=True)
                                with gr.Row():
                                    batch_size = gr.Number(value=5, label="frames per keyframe", precision=1, interactive=True)
                                    fps = gr.Number(value=30, precision=1, label="fps", interactive=True)    
                                    ebsynth_mode = gr.Checkbox(label="EBSynth Mode", value=False)
                                with gr.Row():
                                    savesettings = gr.Button("Save Settings") 
                                with gr.Row():
                                    batch_folder = gr.Textbox(label="Target Folder",placeholder="This is ignored if neither batch run or ebsynth are checked")

                                with gr.Row():
                                    with gr.Accordion("Batch Settings",open=False):
                                        with gr.Row():
                                            batch_checkbox = gr.Checkbox(label="Batch Run", value=False)
                                            max_keyframes = gr.Number(value=-1, label="Max key frames", precision=1, interactive=True,placeholder="for all frames")
                                            border_frames = gr.Number(value=2, label="Border Key Frames", precision=1, interactive=True,placeholder="border frames")
                                with gr.Row():
                                    with gr.Accordion("EBSynth Settings",open=False):
                                        with gr.Row():
                                            split_video = gr.Checkbox(label="Split Video", value=False)
                                            split_based_on_cuts = gr.Checkbox(label="Split based on cuts (as well)", value=False)
                                            #interpolate = gr.Checkbox(label="Interpolate(high memory)", value=False)


            savesettings.click(
                fn=save_settings,
                inputs=[fps,sides,batch_size,video]
            )   
            with gr.Tabs(elemn_id="TemporalKit_gallery_container"):
                with gr.TabItem(elem_id="output_TemporalKit", label="Output"):
                    with gr.Row():
                        result_image = gr.outputs.Image(type='pil')
                    with gr.Row():
                        runbutton = gr.Button("Run") 
                    with gr.Row():
                        send_to_buttons = parameters_copypaste.create_buttons(["img2img"])
                                    
                try:
                    parameters_copypaste.bind_buttons(send_to_buttons, result_image, [""])
                except:
                    print("failed")
                    pass
    parameters_copypaste.add_paste_fields("TemporalKit", result_image,None)
    runbutton.click(preprocess_video, [video,fps,batch_size,sides,resolution,batch_checkbox,max_keyframes,batch_folder,border_frames,ebsynth_mode,split_video,split_based_on_cuts], result_image)


def show_textbox(option):
    if option == True:
        return gr.inputs.Textbox(lines=2, placeholder="Enter your text here")
    else:
        return False

def create_diffusing_tab ():
    global diffuseimg 
    with gr.Column(visible=True, elem_id="Processid") as second_panel:
        dummy_component = gr.Label(visible=False)
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Generate"):
                        with gr.Column():
                            with gr.Row():
                                input_image = gr.Image(label="Input_Image", elem_id="input_page2")
                                input_video = gr.Video(label="Input Video", elem_id="input_videopage2")
                            with gr.Row():
                                read_last_settings = gr.Button("read_last_settings", elem_id="read_last_settings")
                                read_last_image = gr.Button("read_last_image", elem_id="read_last_image")
                            with gr.Row():
                                fps = gr.Number(label="FPS",value=10,precision=1)
                                per_side = gr.Number(label="per side",value=3,precision=1)
                                output_resolution_single = gr.Number(label="output height resolution",value=1024,precision=1)
                                batch_size_diffuse = gr.Number(label="batch size",value=10,precision=1)
                            with gr.Row():
                                runButton = gr.Button("run", elem_id="run_button")

                            
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Output"):
                        with gr.Column():
                            #newbutton = gr.Button("update", elem_id="update_button")
                            outputfile = gr.Video()
    
        read_last_image.click(
        fn=update_image,
        outputs=input_image
        )
        read_last_settings.click(
        fn=update_settings,
        outputs=[fps,per_side,batch_size_diffuse,input_video]
        )
        runButton.click(
        fn=apply_image_to_video,
        inputs=[input_image, input_video,fps,per_side,output_resolution_single,batch_size_diffuse],
        outputs=outputfile
        )


def create_batch_tab ():
    with gr.Column(visible=True, elem_id="batch_process") as second_panel:
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Generate Batch"):
                        with gr.Column():
                            with gr.Row():
                                input_folder = gr.Textbox(label="Input Folder",placeholder="the whole folder, generated before, not just the output folder")
                                input_video = gr.Video(label="Input Video", elem_id="input_videopage2")
                            with gr.Row():
                                read_last_settings = gr.Button("read_last_settings", elem_id="read_last_settings")
                            with gr.Row():
                                fps = gr.Number(label="FPS",value=10,precision=1)
                                per_side = gr.Number(label="per side",value=3,precision=1)
                                output_resolution_batch = gr.Number(label="output resolution",value=1024,precision=1)
                                batch_size = gr.Number(label="batch size",value=5,precision=1)
                                max_frames = gr.Number(label="max frames",value=100,precision=1)
                                border_frames = gr.Number(label="border frames",value=1,precision=1)
                            with gr.Row():
                                runButton = gr.Button("run", elem_id="run_button")
                            

                            
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Output"):
                        with gr.Column():
                            #newbutton = gr.Button("update", elem_id="update_button")
                            outputfile = gr.Video()
    
        read_last_settings.click(
        fn=update_settings_from_file,
        inputs=[input_folder],
        outputs=[fps,per_side,batch_size,input_video,max_frames,border_frames]
        )
        runButton.click(
        fn=apply_image_to_vide_batch,
        inputs=[input_folder,input_video,fps,per_side,output_resolution_batch,batch_size,max_frames,border_frames],
        outputs=outputfile
        )


def create_ebsynth_tab():
    with gr.Column(visible=True, elem_id="batch_process") as second_panel:
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Generate Batch"):
                        with gr.Column():
                            with gr.Row():
                                input_folder = gr.Textbox(label="Input Folder",placeholder="the whole folder, generated before, not just the output folder")
                                input_video = gr.Video(label="Input Video", elem_id="input_videopage2")
                            with gr.Row():
                                read_last_settings_synth = gr.Button("read_last_settings", elem_id="read_last_settings")
                            with gr.Row():
                                fps = gr.Number(label="FPS",value=10,precision=1)
                                per_side = gr.Number(label="per side",value=3,precision=1)
                                output_resolution_batch = gr.Number(label="output resolution",value=1024,precision=1)
                                batch_size = gr.Number(label="batch size",value=5,precision=1)
                                max_frames = gr.Number(label="max frames",value=100,precision=1)
                                border_frames = gr.Number(value=1, label="Border Frames", precision=1, interactive=True,placeholder="border frames")
                            with gr.Row():
                                runButton = gr.Button("prepare ebsynth", elem_id="run_button")
                                recombineButton = gr.Button("recombine ebsynth", elem_id="recombine_button")
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Output"):
                        with gr.Column():
                            #newbutton = gr.Button("update", elem_id="update_button")
                            outputvideo = gr.File()
        read_last_settings_synth.click(
        fn=update_settings_from_file,
        inputs=[input_folder],
        outputs=[fps,per_side,batch_size,input_video,max_frames,border_frames]
        )
        runButton.click(
        fn=post_process_ebsynth,
        inputs=[input_folder,input_video,fps,per_side,output_resolution_batch,batch_size,max_frames,border_frames],
        outputs=outputvideo
        )
        recombineButton.click(
        fn=recombine_ebsynth,
        inputs=[input_folder,fps,border_frames,batch_size],
        outputs=outputvideo
        )
tabs_list = ["TemporalKit"]

def on_ui_tabs():
   
    with gr.Blocks(analytics_enabled=False) as temporalkit:
        with gr.Tabs(elem_id="TemporalKit-Tab") as tabs:
                with gr.Tab(label="Pre-Processing"):
                    with gr.Blocks(analytics_enabled=False):    
                        create_video_Processing_Tab()
                with gr.Tab(label="Temporal-Warp",elem_id="processbutton"):
                    with gr.Blocks(analytics_enabled=False):          
                        create_diffusing_tab()
                with gr.Tab(label="Batch-Warp",elem_id="batch-button"):
                    with gr.Blocks(analytics_enabled=False):          
                        create_batch_tab()
                with gr.Tab(label="Ebsynth-Process",elem_id="Ebsynth-Process"):
                    with gr.Blocks(analytics_enabled=False):          
                        create_ebsynth_tab()
        return (temporalkit, "Temporal-Kit", "TemporalKit"),



def generate(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    negative_prompt: str,
    batch_number: int,
    scale: int,
    batch_in_check,
    batch_in_dir,
    sampler
    ):
        
    model = shared.sd_model
    model.eval().to(shared.device)
    
    animated_gifs = []


def on_ui_settings():
    section = ('TemporalKit', "Temporal-Kit")
    shared.opts.add_option("def_img_cfg", shared.OptionInfo("1.5", "Default Image CFG", section=('ip2p', "Instruct-pix2pix")))



from fastapi import FastAPI, Body
from base64 import b64decode, b64encode
from io import BytesIO

def img_to_b64(image: Image.Image):
    buf = BytesIO()
    image.save(buf, format="png")
    return b64encode(buf.getvalue()).decode("utf-8")

def b64_to_img(enc: str):
    if enc.startswith('data:image'):
        enc = enc[enc.find(',')+1:]
    return Image.open(BytesIO(b64decode(enc)))




script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
