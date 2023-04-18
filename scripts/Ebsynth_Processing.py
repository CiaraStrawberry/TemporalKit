import os
import glob
#om nom nom nom
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
import cv2
import scripts.Berry_Method as bmethod
import scripts.berry_utility as butility
import re



def sort_into_folders(video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_textures,max_frames,output_folder,border):
    per_batch_limmit = (((per_side * per_side) * batch_size)) + border
    video_data = bmethod.convert_video_to_bytes(video_path)
    frames = bmethod.extract_frames_movpie(video_data, fps, max_frames)
    print(f"full frames num = {len(frames)}")


    output_frames_folder = os.path.join(output_folder, "frames")
    output_keys_folder = os.path.join(output_folder, "keys")
    for i, frame in enumerate(frames):
        bmethod.save_square_texture(frame, os.path.join(output_frames_folder, "frames{:05d}.png".format(i)))
    original_frame_height,original_frame_width = frames[0].shape[:2]
    print (original_frame_height,original_frame_width)
    bigbatches,frameLocs = bmethod.split_frames_into_big_batches(frames, per_batch_limmit,border,ebsynth=True,returnframe_locations=True)
    bigprocessedbatches = []


    print (len(square_textures))
    for a,bigbatch in enumerate(bigbatches):
        batches = bmethod.split_into_batches(bigbatches[a], batch_size,per_side* per_side)

        keyframes = [batch[int(len(batch)/2)] for batch in batches]
        if a < len(square_textures):
            new_frames = bmethod.split_square_texture(square_textures[a],len(keyframes), per_side* per_side,_smol_resolution)
            new_frame_start,new_frame_end = frameLocs[a]
            for b in range(len(new_frames)):
                print (new_frame_start)
                inner_start = new_frame_start + (b * batch_size)
                inner_end = new_frame_start + ((b+1) * batch_size)
                frame_position  = inner_start + int((inner_end - inner_start)/2)
                frame_to_save = cv2.resize(new_frames[b], (original_frame_width, original_frame_height), interpolation=cv2.INTER_LINEAR)
                bmethod.save_square_texture(frame_to_save, os.path.join(output_keys_folder, "keys{:05d}.png".format(frame_position)))
    
    just_frame_groups = []
    for i in range(len(bigprocessedbatches)):
        newgroup = []
        for b in range(len(bigprocessedbatches[i])):
            newgroup.append(bigprocessedbatches[i][b])
        just_frame_groups.append(newgroup)

    return


def recombine (video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_textures,max_frames,output_folder,border):
    just_frame_groups = []
    per_batch_limmit = (((per_side * per_side) * batch_size)) - border
    video_data = bmethod.convert_video_to_bytes(video_path)
    frames = bmethod.extract_frames_movpie(video_data, fps, max_frames)
    bigbatches,frameLocs = bmethod.split_frames_into_big_batches(frames, per_batch_limmit,border,returnframe_locations=True)
    bigprocessedbatches = []
    for i in range(len(bigprocessedbatches)):
        newgroup = []
        for b in range(len(bigprocessedbatches[i])):
            newgroup.append(bigprocessedbatches[i][b])
        just_frame_groups.append(newgroup)

    combined = bmethod.merge_image_batches(just_frame_groups, border)

    save_loc = os.path.join(output_folder, "non_blended.mp4")
    generated_vid = bmethod.pil_images_to_video(combined,save_loc, fps)



def crossfade_images(image1, image2, alpha):
    """Crossfade between two images with a given alpha value."""
    image1 = image1.convert("RGBA")
    image2 = image2.convert("RGBA")
    return Image.blend(image1, image2, alpha)

def crossfade_folder_of_folders(output_folder, fps):
    """Crossfade between images in a folder of folders and save the results."""
    root_folder = output_folder
    all_dirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    dirs = [d for d in all_dirs if d.startswith("out_")]

    dirs.sort()

    output_images = []
    allkeynums = getkeynums(os.path.join(root_folder, "keys"))
    print(allkeynums)

    for b in range(allkeynums[0]):
        current_dir = os.path.join(root_folder, dirs[0])
        images_current = sorted(os.listdir(current_dir))
        image1_path = os.path.join(current_dir, images_current[b])
        image1 = Image.open(image1_path)
        output_images.append(np.array(image1))

    for i in range(len(dirs) - 1):
        current_dir = os.path.join(root_folder, dirs[i])
        next_dir = os.path.join(root_folder, dirs[i + 1])

        images_current = sorted(os.listdir(current_dir))
        images_next = sorted(os.listdir(next_dir))

        startnum = get_num_at_index(current_dir,0)
        bigkeynum = allkeynums[i]
        keynum = bigkeynum - startnum
        print(f"recombining directory {dirs[i]} and {dirs[i+1]}, len {keynum}")
        



        for j in range(keynum, len(images_current)):
            alpha = (j - keynum) / (len(images_current) - keynum)
            image1_path = os.path.join(current_dir, images_current[j])
            next_image_index = j - keynum if j - keynum < len(images_next) else len(images_next) - 1
            image2_path = os.path.join(next_dir, images_next[next_image_index])

            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)

            blended_image = crossfade_images(image1, image2, alpha)
            output_images.append(np.array(blended_image))
            # blended_image.save(os.path.join(output_folder, f"{dirs[i]}_{dirs[i+1]}_crossfade_{j:04}.png"))

    final_dir = os.path.join(root_folder, dirs[-1])
    for c in range(allkeynums[-1], len(final_dir)):
        
        images_final = sorted(os.listdir(current_dir))
        image1_path = os.path.join(current_dir, images_final[c])
        image1 = Image.open(image1_path)
        output_images.append(np.array(image1))
    


    output_save_location = os.path.join(output_folder, "crossfade.mp4")
    generated_vid = bmethod.pil_images_to_video(output_images, output_save_location, fps)
    return generated_vid

def getkeynums (folder_path):
    filenames = os.listdir(folder_path)

    # Filter filenames to keep only the ones starting with "keys" and ending with ".png"
    keys_filenames = [f for f in filenames if f.startswith("keys") and f.endswith(".png")]

    # Sort the filtered filenames
    sorted_keys_filenames = sorted(keys_filenames, key=lambda f: int(re.search(r'(\d+)', f).group(0)))

    # Extract the numbers from the sorted filenames
    return [int(re.search(r'(\d+)', f).group(0)) for f in sorted_keys_filenames]


def get_num_at_index(folder_path,index):
    """Get the starting number of the output images in a folder."""
    filenames = os.listdir(folder_path)

    # Filter filenames to keep only the ones starting with "keys" and ending with ".png"
    #keys_filenames = [f for f in filenames if f.startswith("keys") and f.endswith(".png")]

    # Sort the filtered filenames
    sorted_keys_filenames = sorted(filenames, key=lambda f: int(re.search(r'(\d+)', f).group(0)))

    # Extract the numbers from the sorted filenames
    numbers = [int(re.search(r'(\d+)', f).group(0)) for f in sorted_keys_filenames]
    return numbers[index]