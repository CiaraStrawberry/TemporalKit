import os
import glob
import requests
import json
from pprint import pprint
import base64
import numpy as np
from io import BytesIO
import scripts.optical_flow_simple as opflow
from PIL import Image, ImageOps,ImageFilter
import io
from collections import deque
import scripts.berry_utility as utilityb
import cv2
import scripts.optical_flow_raft as raft

# Replace with the actual path to your image file and folder
x_path = "./init.png"
y_folder = "./Input_Images"
temp_folder = "./temp"
frame_count = 0

img2imgurl = "http://localhost:7860/sdapi/v1/img2img"

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists('intensitymaps'):
    os.makedirs('intensitymaps')

if not os.path.exists('temp'):
    os.makedirs('temp')

y_paths = utilityb.get_image_paths(y_folder)


# get the initial image
def square_Image_request (image_path,prompt,denoise_strength,resolution,seed):
    print(len(image_path),prompt,denoise_strength,resolution,seed)
    data = {
        "init_images": [image_path],
        "resize_mode": 0,
        "denoising_strength": denoise_strength,
        "prompt": prompt,
        "negative_prompt": "",
        #"control_net_enabled": "true",
        "alwayson_scripts": {
            "ControlNet":{
                "args": [
                    {
                        "input_image": image_path,
                        "module": "hed",
                       # "model": "control_canny-fp16 [e3fe7712]",
                        "model": "control_hed-fp16 [13fee50b]",
                        "processor_res": 1024,
                        "weight": 1
                    }                  
                ]
            }
        },
        "seed": seed,
        "sampler_index": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 6,
        "width": resolution,
        "height": resolution,
        "restore_faces": True,
        "include_init_images": False,
        "override_settings": {},
        "override_settings_restore_afterwards": True
    }
    response = requests.post(img2imgurl, json=data)
    #print (response.content)
    print(len(json.loads(response.content)["images"]))
    if response.status_code == 200:
        return json.loads(response.content)["images"][0]
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))
            
        except json.JSONDecodeError:
            error_data = response.content
            print(f"Error: Unable to parse JSON error data. {error_data}")
        return None


def prepare_request(allpaths,index,last_stylized,resolution,seed,last_mask,last_last_mask,fillindenoise,edgedenoise,target,diffuse,forwards):
    
    warped_path,flow,unused_mask,whitepixels,flow_img = raft.apply_flow_based_on_images(allpaths[index - 1],allpaths[index],last_stylized,resolution,index,temp_folder)

    #warped_path,flow,unused_mask,whitepixels,flow_img = raft.apply_flow_based_on_images(allpaths[index - 1],allpaths[index],allpaths[index],resolution,index,temp_folder)
    if diffuse:
        hed = gethedfromsd(warped_path,resolution)
        hed = utilityb.mask_to_grayscale( utilityb.scale_mask_intensity(utilityb.base64_to_texture(hed),edgedenoise))
        #hardened_hed = utilityb.harden_mask(hed,True)
    # flow_adjusted_mask = utilityb.modify_intensity_based_on_flow(hardened_hed,flow)
    if last_mask is None:
        last_mask = np.zeros((resolution,resolution))
    if last_last_mask is None:
        last_last_mask = np.zeros((resolution,resolution))
    if diffuse:
        combined_mask = utilityb.combine_masks([unused_mask,utilityb.scale_mask_intensity(last_mask,0.6),utilityb.scale_mask_intensity(last_last_mask,0.3),hed])
    #combine_mask_no_hed = utilityb.combine_masks([unused_mask,utilityb.scale_mask_intensity(last_mask,0.6),utilityb.scale_mask_intensity(last_last_mask,0.3)])
    if whitepixels > 0:
        #replaced = utilityb.replace_masked_area(flow,index,warped_path,unused_mask,warped_path)
        if target is not None and index > 1:
            replaced = utilityb.replaced_mask_from_other_direction_debug(index,Image.open(warped_path).convert("RGBA"),unused_mask,flow,Image.fromarray(cv2.cvtColor(utilityb.base64_to_texture( target), cv2.COLOR_BGR2RGB)).convert("RGBA"),forwards)
        else:
            replaced = utilityb.replaced_mask_from_other_direction_debug(index,Image.open(warped_path).convert("RGBA"),unused_mask,flow,None)
    else:
        replaced = warped_path
    #replaced = replace_masked_area(flow,index,last_stylized,hed,allpaths[index])
    
    
    # Save the mask as a PNG file in the temporary folder
    file_path = os.path.join("debug2", f'mask{index}.png')
    cv2.imwrite(file_path, unused_mask)

    

    if diffuse:
        return send_request_in_chain(last_stylized,allpaths[index],replaced,utilityb.texture_to_base64(combined_mask),index,seed,fillindenoise,resolution),unused_mask,flow_img
    else:
        return replaced, unused_mask,flow_img
   # return send_request(last_stylized,y_folder,allpaths[index],replaced,hed,index)


# Send the request for hed map from the server based on a filepath
def gethedfromsd(image_path,resolution):
    print(resolution)
    #with open(image_path, "rb") as f:
    #    image = base64.b64encode(f.read()).decode("utf-8")
    image_smol = utilityb.resize_base64_image(image_path,resolution,resolution)
    url = "http://127.0.0.1:7860/controlnet/detect"
    data2 = {
        "controlnet_module": "hed",
        "controlnet_input_images": [image_smol],
        "controlnet_processor_res": resolution,
    }
    
    response = requests.post(url, json=data2)
    if response.status_code == 200:
        data = response.content
        loaded = json.loads(data)
        #print(response.content)
        encoded_image = loaded["images"][0]
        #print(f"encoded: {encoded_image}")
        
        return encoded_image
    elif response.status_code == 422:
        error = response.json()
        print("Validation error:", error)
    else:
        print("Unexpected error from hed:", response.status_code, response.text)
        
        



#send based on current situation
def send_request_in_chain(last_image_path,current_image_path,last_warped_path,mask,index,seed,fillindenoise,resolution):

   # with open(last_image_path, "rb") as f:
    #    last_image = base64.b64encode(f.read()).decode("utf-8")
   # last_image = last_image_path
   #with open(current_image_path, "rb") as b:
    #    current_image = base64.b64encode(b.read()).decode("utf-8")
    current_image = current_image_path
    
    if not last_warped_path == "":
        if os.path.isfile(last_warped_path):
            with open(last_warped_path, "rb") as c:
                last_warped = base64.b64encode(c.read()).decode("utf-8")
        else:
            last_warped = last_warped_path
    else:
        last_warped = current_image
    
  


    data = {
        "init_images": [last_warped],
        "inpainting_fill": 1,
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 1,
        "inpainting_mask_invert": 0,
        "resize_mode": 0,
        "denoising_strength": fillindenoise,
        "prompt":"",
        "negative_prompt": "(ugly:1.3), (fused fingers), (too many fingers), (bad anatomy:1.5), (watermark:1.5), (words), letters, untracked eyes, asymmetric eyes, floating head, (logo:1.5), (bad hands:1.3), (mangled hands:1.2), (missing hands), (missing arms), backward hands, floating jewelry, unattached jewelry, floating head, doubled head, unattached head, doubled head, head in body, (misshapen body:1.1), (badly fitted headwear:1.2), floating arms, (too many arms:1.5), limbs fused with body, (facial blemish:1.5), badly fitted clothes, imperfect eyes, untracked eyes, crossed eyes, hair growing from clothes, partial faces, hair not attached to head",
        "alwayson_scripts": {
            "ControlNet":{
                "args": [
                    {
                        "input_image": current_image,
                        "module": "hed",
                        "model": "control_hed-fp16 [13fee50b]",
                        "weight": 1,
                        "guidance": 1,
                    },
                    {
                        "input_image": last_warped,
                        "model": "diff_control_sd15_temporalnet_fp16 [adc6bd97]",
                        "module": "none",
                        "weight": 1,
                        "guidance": 1,
                    }
                  
                ]
            }
        },
        "seed": seed,
        "subseed": -1,
        "subseed_strength": -1,
        "sampler_index": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 6,
        "width": resolution,
        "height": resolution,
        "restore_faces": True,
        "include_init_images": True,
        "override_settings": {},
        "override_settings_restore_afterwards": True
    }
    
    if not mask == "":
        data["mask"] = mask
        if not os.path.exists("./debug2/"):
            os.makedirs("./debug2/")
        with open(f"./debug2/{index}.png", "wb") as e:
            e.write(base64.b64decode(mask))
            print(f"debug mask saved at ./debug2/{index}.png")
    else:
        data['denoising_strength'] = 0
    response = requests.post(img2imgurl, json=data)
   # print (response.content)
    print(response.status_code)
    if response.status_code == 200:
        return json.loads(response.content)["images"][0]
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))
            
        except json.JSONDecodeError:
            error_data = response.content
            print(f"Error: Unable to parse JSON error data. {error_data}")
        return None


def batch_sd_run (y_paths, initial,count,seed,skip_first,fillindenoise,edgedenoise,smol_resolution,Forwards,target,diffuse):
    output_images = []
    all_flow = []
    output_images.append(initial)
    last_mask = None
    last_last_mask = None
    allpaths = y_paths
    for i in range(1, len(y_paths)):
        current_frame = count + i
        result,mask,flow = prepare_request(allpaths,i,output_images[i-1],smol_resolution,seed,last_mask,last_last_mask,fillindenoise,edgedenoise,target,diffuse,Forwards)
        all_flow.append(flow)
        output_images.append(result)
        print(f"Written data for frame {current_frame}:")
        last_last_mask = last_mask
        last_mask = mask
    if (skip_first == True):
        output_images.pop(0)        


    
    return output_images,all_flow


#output_images = []
#datanew = send_request_in_chain(y_paths[0], x_path,"","",0)
#output_images.append(datanew)
#output_paths = []

#for i in range(1, len(y_paths)):
#     frame_count = frame_count + 1
#     result_image = output_images[i-1]
#     temp_image_path = os.path.join(output_folder, f"temp_image_{i}.png")
#     data = json.loads(result_image)
#     encoded_image = data["images"][0]
#     with open(temp_image_path, "wb") as f:
#         f.write(base64.b64decode(encoded_image))
#     output_paths.append(temp_image_path)
#     #result = send_request(temp_image_path, y_folder, y_paths[i])
#     result = prepare_request(y_paths,i,temp_image_path,512)
#     output_images.append(result)
#     print(f"Written data for frame {i}:")


