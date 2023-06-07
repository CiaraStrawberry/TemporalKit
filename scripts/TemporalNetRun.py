import numpy as np
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import write_jpeg
import torchvision.transforms as T
import scripts.berry_utility as utilityb
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from scipy.interpolate import LinearNDInterpolator
from imageio import imread, imwrite
from torchvision.utils import flow_to_image
import os
import glob
import requests
import json
import numpy as np
import sys
import torch
from PIL import Image
from pprint import pprint
import base64
from io import BytesIO
import cv2
import pickle
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()



def get_image_resolution(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get image resolution
        width, height = img.size
    return width, height

def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    
    def extract_number(file_name):
        # Extract the numbers from the filename
        numbers = re.findall(r'\d+', os.path.basename(file_name))
        numbers = [int(num) for num in numbers]
        # If no numbers are found, return a tuple that will sort last
        if not numbers:
            return (float('inf'), float('inf'))
        # If only one number is found, assume it's the second number and default the first to infinity
        elif len(numbers) == 1:
            return (float('inf'), numbers[0])
        else:
            return tuple(numbers)

    return sorted(files, key=extract_number)

def run_stable_diffusion(project_folder,init_image,width,height,positive_prompt,negative_prompt,denoise):
    output_images = []
    output_paths = []

    if (width == -1):
        width, height = get_image_resolution(init_image)
    temp_folder = os.path.join(project_folder, "Temp")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    # Initialize with the first image path
    input_folder = os.path.join(project_folder, "Input")
    output_folder = os.path.join(project_folder, "Output")
    #output_image_path = os.path.join(output_folder, f"output_image_0.png")
    y_paths = get_image_paths(input_folder)
    init_image_path = os.path.join(input_folder, f"init_image.png")
   # with open(init_image_path, "wb") as f:
    #    f.write(init_image)
    print (init_image)
    result = init_image
    last_image_path = result
    for i in range(1, len(y_paths)):
        # Use the last image path and optical flow map to generate the next input
        optical_flow = infer(i,temp_folder,y_paths[i - 1], y_paths[i],width,height)
        
        # Modify your send_request to use the last_image_path
        result = send_request(init_image,last_image_path, optical_flow, y_paths[i],width,height,positive_prompt,negative_prompt,denoise)
        data = json.loads(result)
        encoded_image = data["images"][0]
        output_image_path = os.path.join(output_folder, f"output_image_{i}.png")
        last_image_path = output_image_path
        with open(output_image_path, "wb") as f:
            f.write(base64.b64decode(encoded_image))
        print(f"Written data for frame {i}:")



def send_request(init_image_path,last_image_path, optical_flow_path,current_image_path,width,height,positive_prompt,negative_prompt,denoise):
    url = "http://localhost:7860/sdapi/v1/img2img"
    
    with open(last_image_path, "rb") as b:
       last_image_encoded = base64.b64encode(b.read()).decode("utf-8")
    
    # Load and process the last image
    last_image = cv2.imread(last_image_path)
    last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)
    last_image = cv2.resize(last_image, (width, height))

    # Load and process the optical flow image
    flow_image = cv2.imread(optical_flow_path)
    flow_image = cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB)

    # Load and process the current image
    with open(current_image_path, "rb") as b:
       current_image = base64.b64encode(b.read()).decode("utf-8")
    
    with open(init_image_path, "rb") as c:
        init_image = base64.b64encode(c.read()).decode("utf-8")

    # Concatenating the three images to make a 6-channel image
    six_channel_image = np.dstack((last_image, flow_image))

    # Serializing the 6-channel image
    serialized_image = pickle.dumps(six_channel_image)

    # Encoding the serialized image
    encoded_image = base64.b64encode(serialized_image).decode('utf-8')

    data = {
        "init_images": [current_image],
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 1,
        "inpainting_mask_invert": 1,
        "resize_mode": 0,
        "denoising_strength": denoise,
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "alwayson_scripts": {
            "ControlNet":{
                "args": [
                    {
                        "input_image": current_image,
                        "module": "hed",
                        "model": "control_hed-fp16 [13fee50b]",
                        "weight": 0.7,
                        "guidance": 1,
                   },
                    {
                        "input_image": encoded_image,
                        "model": "temporalnetversion2 [b146ac48]",
                        "module": "none",
                        "weight": 0.6,
                        "guidance": 1,
                    },
                    {
                        "input_image": current_image,
                        "model": "control_v11p_sd15_openpose [cab727d4]",
                        "module": "openpose_full",
                        "weight": 0.7,
                        "guidance":1,
                    }
                    
                  
                ]
            }
        },
        "seed": 4123457655,
        "subseed": -1,
        "subseed_strength": -1,
        "sampler_index": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 6,
        "width": 512,
        "height": 512,
        "restore_faces": True,
        "include_init_images": True,
        "override_settings": {},
        "override_settings_restore_afterwards": True
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))
            
        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON error data.")
        return None



def infer(i,output_folder,frameA, frameB,width,height):
    #video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    #video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
    #_ = urlretrieve(video_url, video_path)
   
    #frames, _, _ = read_video(str("./spacex.mp4"), output_format="TCHW")
    #print(f"FRAME BEFORE stack: {frames[100]}")
    
    
    input_frame_1 = read_image(str(frameA), ImageReadMode.RGB)
   
    input_frame_2 = read_image(str(frameB), ImageReadMode.RGB)
 
    
    #img1_batch = torch.stack([frames[0]])
    #img2_batch = torch.stack([frames[1]])

    img1_batch = torch.stack([input_frame_1])
    img2_batch = torch.stack([input_frame_2])
    
    
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()


    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[width, height])
        img2_batch = F.resize(img2_batch, size=[width, height])
        return transforms(img1_batch, img2_batch)


    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)


    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

    predicted_flows = list_of_flows[-1]


    #flow_imgs = flow_to_image(predicted_flows)

    #print(flow_imgs)

    predicted_flow = list_of_flows[-1][0]
    opitcal_flow_path = os.path.join(output_folder, f"flow_{i}.png")
    flow_img = flow_to_image(predicted_flow).to("cpu")
    write_jpeg(flow_img,opitcal_flow_path)
    
    
    return opitcal_flow_path

