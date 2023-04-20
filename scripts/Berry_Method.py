import cv2
import base64
from PIL import Image
import numpy as np
import math
import scripts.berry_utility as utilityb
import scripts.stable_diffusion_processing as sdprocess
from moviepy.editor import *
import os
import subprocess
import json
import tempfile
import uuid
from torchvision.io import write_jpeg
import re
import gradio as gr
from io import BytesIO
import shutil

resolution = 1400
smol_resolution = 512
prompt = "cyborg humans photo realistic"
fill_in_denoise = 0
edge_denoise = 0.4 # this is a factor of fill in denoise
initial_denoise = 0.85
frames_limit = 50
seed = 5434536443
video_path = "./pexels-alena-darmel-7184187_resized.mp4"
diffuse = False
check_edges = False

def split_into_batches(frames, batch_size, max_batches):
    groups = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)][:max_batches]

    # Add any remaining images to the last group
    if len(frames) > max_batches * batch_size:
        groups[-1] += frames[max_batches*batch_size:]

    return groups

def create_square_texture(frames, max_size, side_length=3):

    original_height, original_width = frames[0].shape[:2]
    # Calculate the average aspect ratio of the input frames
    big_frame_width = original_width * side_length
    big_frame_height = original_height * side_length

    texture_aspect_ratio = float(big_frame_width) / float(big_frame_height)
    _smol_frame_height = max_size
    _smol_frame_width = int(_smol_frame_height * texture_aspect_ratio)


    actual_texture_width, actual_texture_height = resize_to_nearest_multiple(_smol_frame_width, _smol_frame_height, side_length)

    frames_per_row = side_length
    frame_width = int (actual_texture_width / side_length)
    frame_height = int(actual_texture_height / side_length)
    print (f"generating square of height {actual_texture_width} and width {actual_texture_height}")

    texture = np.zeros((actual_texture_height, actual_texture_width, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        if frame is not None and not frame.size == 0:
            resized_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            row, col = i // frames_per_row, i % frames_per_row
            texture[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = resized_frame

    return texture

def split_frames_into_big_batches(frames, batch_size, border,ebsynth,returnframe_locations=False):
    """
    Splits an array of numpy frames into batches of a given size, adding a certain number of border
    frames from the next batch to each batch.

    Parameters:
    frames (numpy.ndarray): The input frames to be split.
    batch_size (int): The number of frames per batch.
    border (int): The number of border frames from the next batch to add to each batch.

    Returns:
    List[numpy.ndarray]: A list of batches, each containing `batch_size` + `border` frames (except for the last batch).
    """
    num_frames = len(frames)
    num_batches = int(np.ceil(num_frames / batch_size))
    print(f"frames num = {len(frames)} while num batches = {num_batches}")
    batches = []

    frame_locations = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if ebsynth == False:
            # Add border frames if not the last batch and if available
            if i < num_batches - 1:
                end_idx += min(border, num_frames - end_idx)
            else:
                # Combine the last batch with the previous batch if the number of frames in the last batch is smaller than the border size
                if end_idx - start_idx < border and len(batches) > 0:
                    batches[-1] = np.concatenate((batches[-1], frames[start_idx:end_idx]))
                    break
        else:
            if i < num_batches - 1:
                end_idx = end_idx + border

        end_idx = min(end_idx, num_frames)
        batches.append(frames[start_idx:end_idx])
        frame_locations.append((start_idx,end_idx))

    if returnframe_locations == False:
        return batches
    else:
        return batches,frame_locations

def split_square_texture(texture, num_frames,max_frames, _smol_resolution,ebsynth=False):
    
    texture_height, texture_width = texture.shape[:2]
    texture_aspect_ratio = float(texture_width) / float(texture_height)
    
    frames_per_row = int(math.ceil(math.sqrt(max_frames)))
    frame_height = int (texture_height / frames_per_row)
    frame_width = int(texture_width / frames_per_row)
    
    _smol_frame_height = _smol_resolution
    _smol_frame_width = int(_smol_frame_height * texture_aspect_ratio)

    if ebsynth == False:
        _smol_frame_resized_width, _smol_frame_resized_height = resize_to_nearest_multiple_of_8(_smol_frame_width, _smol_frame_height)
    else:
        _smol_frame_resized_width, _smol_frame_resized_height  = _smol_frame_width, _smol_frame_height 
    #_smol_frame_resized_width, _smol_frame_resized_height = _smol_frame_width, _smol_frame_height
    frames = []

    for i in range(num_frames):
        row, col = i // frames_per_row, i % frames_per_row
        frame = texture[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width]

        if not frame.size == 0:
            resized_frame = cv2.resize(frame, (_smol_frame_resized_width, _smol_frame_resized_height), interpolation=cv2.INTER_AREA)
            frames.append(resized_frame)
        else:
            print("frame size 0")
            frames.append(np.zeros((_smol_frame_resized_width, _smol_frame_resized_height, 3), dtype=np.uint8))

    return frames

def save_square_texture(texture, file_path):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(texture)
    #image = Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB))
    # Save the image to the specified file path
    print(f'saved to {file_path}')
    image.save(file_path, format="PNG")

def convert_video_to_bytes(input_file):
    # Read the uploaded video file
    print(f"reading video file... {input_file}")
    with open(input_file, "rb") as f:
        video_bytes = f.read()

    # Return the processed video bytes (or any other output you want)
    return video_bytes



def generate_square_from_video(video_path, fps, batch_size,resolution,size_size):
    video_data = convert_video_to_bytes(video_path)
    frames_limit = (size_size * size_size) * batch_size
    frames = extract_frames_movpie(video_data, fps, frames_limit)
    print(len(frames))
    number_of_batches = size_size * size_size
    batches = split_into_batches(frames, batch_size,number_of_batches)
    print("Number of batches:", len(batches))
    first_frames = [batch[0] for batch in batches]  
    
    square_texture = create_square_texture(first_frames, resolution,side_length=size_size)
    #save_square_texture(square_texture, "./result/original.png")
    
    return square_texture

def generate_squares_to_folder (video_path, fps, batch_size,resolution,size_size,max_frames,output_folder,border,ebsynth_mode):

    if ebsynth_mode == False:
        if border >=  (batch_size * size_size * size_size) / 2:
            raise Exception("too many border frames, reduce border or increase batch size")
     

    input_folder_loc = os.path.join(output_folder, "input")
    output_folder_loc = os.path.join(output_folder, "output")
    debug_result = os.path.join(output_folder, "result")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(input_folder_loc):
        os.makedirs(input_folder_loc)
    if not os.path.exists(output_folder_loc):
        os.makedirs(output_folder_loc)
    if not os.path.exists(debug_result):
        os.makedirs(debug_result)
    frames_loc = os.path.join(output_folder, "frames")
    keys_loc = os.path.join(output_folder, "keys")

    if ebsynth_mode == True:
        if not os.path.exists(frames_loc):
            os.makedirs(frames_loc)
        if not os.path.exists(keys_loc):
            os.makedirs(keys_loc)

    video_data = convert_video_to_bytes(video_path)
    per_batch_limmit = ((size_size * size_size) * batch_size) + border
    #if ebsynth_mode == False:
    #    per_batch_limmit = per_batch_limmit + border
    frames = extract_frames_movpie(video_data, fps, max_frames)


    bigbatches = split_frames_into_big_batches(frames, per_batch_limmit,border,ebsynth=ebsynth_mode)
    square_textures = []
    height = 0
    width = 0
    for i in range(len(bigbatches)):
        batches = split_into_batches(bigbatches[i], batch_size, size_size * size_size)
        print("Number of batches:", len(batches))
        if ebsynth_mode == False:
            keyframes = [batch[0] for batch in batches]  
        else: 
            keyframes = [batch[int(len(batch)/2)] for batch in batches]
        
        square_texture = create_square_texture(keyframes, resolution,side_length=size_size)
        save_square_texture(square_texture, os.path.join(input_folder_loc, f"input{i}.png"))
        square_textures.append(square_texture)
        height = square_texture.shape[0]
        width = square_texture.shape[1]
    
    batch_settings_loc = os.path.join(output_folder, "batch_settings.txt")
    with open(batch_settings_loc, "w") as f:
        f.write(str(fps) + "\n")
        f.write(str(size_size) + "\n")
        f.write(str(batch_size) + "\n")
        f.write(str(video_path) + "\n")
        f.write(str(max_frames) + "\n")
        f.write(str(border) + "\n")
    #return list of urls
    return square_texture


def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def blend_images(img1, img2, alpha=0.5):
    blended = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    return blended

def resize_to_nearest_multiple(width, height, a):
    def nearest_multiple(n, a, b):
        multiple_a = round(n / a) * a
        multiple_b = round(n / b) * b
        return min(multiple_a, multiple_b, key=lambda x: abs(x - n))

    new_width = nearest_multiple(width, a, 8)
    new_height = nearest_multiple(height, a, 8)
    return int(new_width), int(new_height)

def resize_to_nearest_multiple_of_8(width, height):
    def nearest_multiple(n, factor):
        return round(n / factor) * factor

    new_width = nearest_multiple(width, 8)
    new_height = nearest_multiple(height, 8)
    return new_width, new_height


def merge_image_batches(image_batches, border):
    merged_batches = []
    height, width = image_batches[0][0].shape[:2]

    for i in range(len(image_batches) - 1):
        current_batch = image_batches[i]
        next_batch = image_batches[i + 1]
        for i in range(len(current_batch)):
            current_batch[i] = cv2.resize(current_batch[i], (width, height))
        for i in range(len(next_batch)):
            next_batch[i] = cv2.resize(next_batch[i], (width, height))

        # If it's not the first batch, remove the blended images from the current batch
        if i > 0:
            current_batch = current_batch[border:]

        # Copy all images except the border ones from the current batch
        for j in range(len(current_batch) - border):
            merged_batches.append(current_batch[j])

        # Blend the border images between the current and next batch
        for j in range(border):
            try:
                alpha = float(j) / float(border)
                blended_image = cv2.addWeighted(current_batch[len(current_batch) - border + j], 1 - alpha, next_batch[j], alpha, 0)
                merged_batches.append(blended_image)
            except IndexError:
                print ("merge failed")

    # Add remaining images from the last batch
    merged_batches.extend(image_batches[-1][border:])

    return merged_batches

def process_video_batch (video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_textures,max_frames,output_folder,border):
    
    per_batch_limmit = (((per_side * per_side) * batch_size)) + border
    video_data = convert_video_to_bytes(video_path)
    frames = extract_frames_movpie(video_data, fps, max_frames)
    print(f"full frames num = {len(frames)}")
    bigbatches = split_frames_into_big_batches(frames, per_batch_limmit,border,ebsynth=False)
    bigprocessedbatches = []
    for i , batch in enumerate(bigbatches):
        if i < len(square_textures):
            new_batch = process_video(batch, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_textures[i])
            bigprocessedbatches.append(new_batch)
            for a, image in enumerate(new_batch):
                Image.fromarray(image).save(os.path.join(output_folder, f"result/output{a + (len(new_batch) * i)}.png"))
    
    just_frame_groups = []
    for i in range(len(bigprocessedbatches)):
        newgroup = []
        for b in range(len(bigprocessedbatches[i])):
            newgroup.append(bigprocessedbatches[i][b])
        just_frame_groups.append(newgroup)

    combined = merge_image_batches(just_frame_groups, border)

    save_loc = os.path.join(output_folder, "blended.mp4")
    generated_vid = pil_images_to_video(combined,save_loc, fps)
    return generated_vid


def process_video_single(video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_texture):
    extension_path = os.path.abspath(__file__)
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    output_folder = os.path.join(extension_dir, "result")
    extension_path = os.path.abspath(__file__)
    frames_limit = (per_side * per_side) * batch_size
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    extension_save_folder = os.path.join(extension_dir, "result")
    if not os.path.exists(extension_save_folder):
        os.makedirs(extension_save_folder)
    delete_folder_contents(extension_save_folder)
    #rerun the generatesquarefromvideo function to get the unaltered square texture
    video_data = convert_video_to_bytes(video_path)
    frames = extract_frames_movpie(video_data, fps, frames_limit)
    processed_frames = process_video(frames,per_side,batch_size,fillindenoise,edgedenoise,_smol_resolution,square_texture)
    output_video_path = os.path.join(output_folder, "output.mp4")
    generated_video =  pil_images_to_video(processed_frames, output_video_path, fps)
    return generated_video


def process_video(frames, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_texture):

    frame_count = 0
    print(len(frames))
    batches = split_into_batches(frames, batch_size, per_side * per_side)
    print("Number of batches:", len(batches))
    first_frames = [batch[0] for batch in batches]
    #actuallyprocessthevideo
    debug = False

    frame_count = 0
    global resolution
    print(len(frames))


    if debug is False:
        #result_texture = sdprocess.square_Image_request(encoded_square_texture, prompt, initial_denoise, resolution, seed)
        result_texture = square_texture
        #save_square_texture(encoded_returns, "./result/processed.png")
        
    else:
        f = open("./result/processed.png", "rb")
        bytes = f.read()
        result_texture = base64.b64encode(bytes).decode("utf-8")
        resolution_get = Image.open("./result/processed.png")
        resolution= resolution_get.height
        # this is stupid and inefficiant i dont care
    #its not really encoded anymore is it
    encoded_returns = result_texture
    #encoded_returns = cv2.cvtColor(utilityb.base64_to_texture(result_texture), cv2.COLOR_BGR2RGB)

    new_frames = split_square_texture(encoded_returns,len(first_frames), per_side * per_side,_smol_resolution,False)
    
    if check_edges:
        for i, image in enumerate(new_frames):
            image = utilityb.check_edges(image)
    #:( 

    # Save first frames
    #for idx, first_frame in enumerate(first_frames):
    #    save_square_texture(first_frame, os.path.join(output_folder, f"first_frame_{idx}.png"))
    #    save_square_texture(new_frames[idx], os.path.join(output_folder, f"first_frame_processed_{idx}.png"))
    """
    Turns out merging each frame backwards and forwards doesn't actually work, you'd think it did because each frame is conceptually closer to it's origin, but it breaks the flowmap in all sorts of weird ways if you tell it to go backwards, very very annoying
    last_processed = None
    for i, batch in enumerate(batches):

        encoded_batch = []
        for b, image in enumerate(batch):
            encoded_batch.append(utilityb.texture_to_base64(image))
        encoded_new_frame = utilityb.texture_to_base64(new_frames[i])

        processed_batch_before,all_flow_before = sdprocess.batch_sd_run(encoded_batch, encoded_new_frame, frame_count, seed, False, fillindenoise, edgedenoise, _smol_resolution,False,encoded_new_frame,False)

        if i < len(batches) - 1:

            encoded_batch_next = []
            for b, image in enumerate(batches[i+1]):
                encoded_batch_next.append(utilityb.texture_to_base64(image))
            encoded_ext_frame = utilityb.texture_to_base64(new_frames[i + 1])
            encoded_batch.insert(0,utilityb.texture_to_base64(first_frames[i]))
            processed_batch_after,all_flow_after = sdprocess.batch_sd_run(encoded_batch_next, encoded_new_frame, frame_count, seed, True, fillindenoise, edgedenoise, _smol_resolution,True,encoded_new_frame,False)
            processed_batch_after.append(encoded_ext_frame)

        if last_processed is not None:
            print (len(last_processed))
            print (len(processed_batch_before))
            blended_frames = blend_batches(last_processed, processed_batch_before, resolution=_smol_resolution)
            for b, blended_frame in enumerate(blended_frames):
                savepath = os.path.join(output_folder, f"frame_{frame_count + b}.png")
                #save_square_texture(blended_frame, savepath)
                save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(processed_batch_before[b]), cv2.COLOR_BGR2RGB), savepath)



                #save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(last_processed[b]), cv2.COLOR_BGR2RGB), f"./debug/before_{frame_count + b}.png")
                #save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(processed_batch_before[b]), cv2.COLOR_BGR2RGB), f"./debug/after_{frame_count + b}.png")
            #for c, flow in enumerate(all_flow_before):
                #write_jpeg(flow, f"./debug/before_flow_{frame_count + c + 1}.png")
        else:
            for b, frame in enumerate(processed_batch_before):
                savepath = os.path.join(output_folder, f"frame_{frame_count + b}.png")
                save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB), savepath)


        frame_count += len(batch)
        last_processed = processed_batch_after




    """

    output_pil_images = []
    last_processed = None
    for i, batch in enumerate(batches):

        encoded_new_frame = utilityb.texture_to_base64(new_frames[i])


        encoded_batch = []
        for b, image in enumerate(batches[i]):
            encoded_batch.append(utilityb.texture_to_base64(image))
        
        processed_batch,all_flow_after = sdprocess.batch_sd_run(encoded_batch, encoded_new_frame, frame_count, seed, False, fillindenoise, edgedenoise, _smol_resolution,True,encoded_new_frame,False)
 
        if last_processed is not None:
            encoded_batch.insert(0,utilityb.texture_to_base64(batches[i - 1][-1]))
            processed_batch_from_before,all_flow_after = sdprocess.batch_sd_run(encoded_batch, last_processed, frame_count, seed, True, fillindenoise, edgedenoise, _smol_resolution,True,last_processed,False)
            if not min(len (processed_batch_from_before),len(processed_batch)) > 1:
                return output_pil_images
            blended_frames = blend_batches(processed_batch_from_before, processed_batch, resolution=_smol_resolution)
            for b, blended_frame in enumerate(blended_frames):
                #savepath = os.path.join(output_folder, f"frame_{frame_count + b}.png")
                #save_square_texture(blended_frame, savepath)
                output_pil_images.append(blended_frame)
                #save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(processed_batch_before[b]), cv2.COLOR_BGR2RGB), savepath)



                #save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(last_processed[b]), cv2.COLOR_BGR2RGB), f"./debug/before_{frame_count + b}.png")
                #save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(processed_batch_before[b]), cv2.COLOR_BGR2RGB), f"./debug/after_{frame_count + b}.png")
            #for c, flow in enumerate(all_flow_before):
                #write_jpeg(flow, f"./debug/before_flow_{frame_count + c + 1}.png")
        else:
            for b, frame in enumerate(processed_batch):
                #savepath = os.path.join(output_folder, f"frame_{frame_count + b}.png")
                #save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB), savepath)
                output_pil_images.append(cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB))


        frame_count += len(batch)
        last_processed = processed_batch[-1]

    return output_pil_images

    
def pil_images_to_video(images, output_file, fps=24):
    """
    Turns a list of PIL images into a video using MoviePy.

    :param images: list, a list of PIL Image objects
    :param output_file: str, path to the output video file (e.g., 'output.mp4')
    :param fps: int, frames per second (default: 24)
    """

    

    pil_images = []
    for image in images:
        newimg = Image.fromarray(image)
        pil_images.append(newimg)
    video_size = pil_images[0].size
    better_pil_images = []
    for image in pil_images:
        better_pil_images.append(image.resize(video_size))
            
    
    # Create a temporary directory to store the image files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save images as temporary files
        image_paths = []
        for i, image in enumerate(better_pil_images):
            temp_image_path = os.path.join(temp_dir, f'image_{i:04d}.png')
            image.save(temp_image_path, format='PNG')
            image_paths.append(temp_image_path)

        # Create a clip from the image sequence
        clip = ImageSequenceClip(image_paths, fps=fps)

        # Write the clip to a video file
        clip.write_videofile(output_file, codec='libx264')

    return output_file
    

def image_folder_to_video(folder_path, output_file, fps=24):
    """
    Turns a folder of images into a video using MoviePy.

    :param folder_path: str, path to the folder containing the images
    :param output_file: str, path to the output video file (e.g., 'output.mp4')
    :param fps: int, frames per second (default: 24)
    """
    # Get a list of image file names
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Sort image files by their names
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Create a list of full image file paths
    image_paths = [os.path.join(folder_path, image) for image in image_files]

    # Create a clip from the image sequence
    clip = ImageSequenceClip(image_paths, fps=fps)

    # Write the clip to a video file
    clip.write_videofile(output_file, codec='libx264')

    return output_file



def extract_frames_movpie(input_video, target_fps, max_frames=None):

    def get_video_info(video_path):
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout)

    def create_temp_file(suffix):
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4().hex}{suffix}"
        return os.path.join(temp_dir, temp_filename)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(input_video)
    video_path = f.name


    video_info = get_video_info(video_path)
    video_stream = next((stream for stream in video_info['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    video_clip = VideoFileClip(video_path)
    video_resampled = video_clip.set_fps(target_fps)

    if max_frames is not None and max_frames > 0:
        total_frames = video_resampled.duration * target_fps
        if total_frames > max_frames:
            end_time = max_frames / target_fps
            video_resampled = video_resampled.subclip(0, end_time)

    frames = []
    for frame in video_resampled.iter_frames(dtype="uint8"):
        frames.append(np.array(frame))

    return frames


def blend_batches(batch_before, current_batch,resolution, blend_start_ratio=0.9, blend_end_ratio=0.1):
    blended_frames = []
    num_frames = min(len (batch_before),len(current_batch))
    decoded_batch_before = [cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB) for frame in batch_before]
    decoded_current_batch = [cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB) for frame in current_batch]

    #target_width, target_height = resolution, resolution
    height, width = decoded_batch_before[0].shape[:2]
    # Resize the images in decoded_batch_before and decoded_current_batch
    decoded_batch_before = [cv2.resize(img, (width,height)) for img in decoded_batch_before]
    decoded_current_batch = [cv2.resize(img,  (width,height)) for img in decoded_current_batch]

    output_folder = "moretemp"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not num_frames > 1:
        return
    for i in range(num_frames):
        alpha = blend_start_ratio - (i / (num_frames - 1)) * (blend_start_ratio - blend_end_ratio)
        blended_frame = cv2.addWeighted(decoded_batch_before[i], alpha, decoded_current_batch[i], 1 - alpha, 0)
        print(f"blended frame {i}")

        # Concatenate the two input frames and the blended frame horizontally
        concatenated_frame = cv2.hconcat([decoded_batch_before[i], decoded_current_batch[i], blended_frame])

        cv2.imwrite(os.path.join(output_folder, f"concatenated_{i}.png"), concatenated_frame)

        blended_frames.append(blended_frame)
    return blended_frames



# Example usage:
#if __name__ == "__main__":
#    process_video(video_path, fps=10, frames_limit=45, batch_size=5,fillindenoise=fill_in_denoise,edgedenoise=edge_denoise,_smol_resolution=smol_resolution)



def interpolate_frames(frame1, frame2, alpha):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    h, w = flow.shape[:2]

    flow_map = -alpha * flow + np.indices((h, w)).transpose(1, 2, 0)
    flow_map = flow_map.astype(np.float32)  # Convert flow_map to float32 data type
    return cv2.remap(frame1, flow_map, None, cv2.INTER_LINEAR)

def interpolate_video(input_path, output_path, output_fps):
    clip = VideoFileClip(input_path)
    input_fps = clip.fps

    frames = [frame for frame in clip.iter_frames()]
    new_frames = []

    if output_fps <= input_fps:
        raise ValueError("Output fps should be greater than input fps")

    frame_ratio = input_fps / output_fps

    for i in range(len(frames) - 1):
        new_frames.append(frames[i])
        print(f"interpolating for frame {i}")
        extra_frames = int(round((i + 1) / frame_ratio) - round(i / frame_ratio))
        for j in range(1, extra_frames + 1):
            
            alpha = j / (extra_frames + 1)
            frame1 = frames[i]  # Transpose the dimensions of frame1 (HxWxC to WxHxC)
            frame2 = frames[i + 1]  # Transpose the dimensions of frame2 (HxWxC to WxHxC)
            interpolated_frame = interpolate_frames(frame1, frame2, alpha)
            interpolated_frame = interpolated_frame.transpose(1, 0, 2)  # Transpose back the dimensions of interpolated_frame (WxHxC to HxWxC)
            new_frames.append(interpolated_frame)

    new_frames.append(frames[-1])

    new_clip = ImageSequenceClip(new_frames, fps=output_fps)
    new_clip.write_videofile(output_path)
    return output_path
