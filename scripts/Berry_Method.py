import base64
import math
import re

import cv2
import os
import gradio
import numpy as np
import scripts.berry_utility as utilityb
import scripts.stable_diffusion_processing as sdprocess
from PIL import Image
from moviepy.editor import *
from tqdm.auto import tqdm

resolution = 1400
smol_resolution = 512
prompt = "cyborg humans photo realistic"
fill_in_denoise = 0
edge_denoise = 0.4  # this is a factor of fill in denoise
initial_denoise = 0.85
frames_limit = 50
seed = 5434536443
diffuse = False
check_edges = False


def split_into_batches(frames, batch_size, max_batches):
    groups = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)][:max_batches]

    # Add any remaining images to the last group
    if len(frames) > max_batches * batch_size:
        groups[-1] += frames[max_batches * batch_size:]

    return groups


def create_square_texture(frames, max_size, side_length=3):
    original_height, original_width = frames[0].shape[:2]
    # Calculate the average aspect ratio of the input frames
    big_frame_width = original_width * side_length
    big_frame_height = original_height * side_length

    texture_aspect_ratio = float(big_frame_width) / float(big_frame_height)
    _smol_frame_height = max_size
    _smol_frame_width = int(_smol_frame_height * texture_aspect_ratio)

    actual_texture_width, actual_texture_height = utilityb.resize_to_nearest_multiple(_smol_frame_width,
                                                                                      _smol_frame_height, side_length)

    frames_per_row = side_length
    frame_width = int(actual_texture_width / side_length)
    frame_height = int(actual_texture_height / side_length)
    tqdm.write(f"Generating square of width {actual_texture_width} and height {actual_texture_height}")

    texture = np.zeros((actual_texture_height, actual_texture_width, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        if frame is not None and not frame.size == 0:
            resized_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            row, col = i // frames_per_row, i % frames_per_row
            texture[row * frame_height:(row + 1) * frame_height,
            col * frame_width:(col + 1) * frame_width] = resized_frame
            # truth be told i am not entirely sure why this is needed
            fixed_texture = cv2.resize(texture, (actual_texture_width, actual_texture_height),
                                       interpolation=cv2.INTER_AREA)

    return fixed_texture


def split_frames_into_big_batches(frames, batch_size, border, ebsynth, returnframe_locations=False):
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
    tqdm.write(f"Frames num = {len(frames)} while num batches = {num_batches}")
    batches = []

    frame_locations = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if not ebsynth:
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
        tqdm.write(f"Batch {i} has {len(batches[i])} frames")
        frame_locations.append((start_idx, end_idx))

    if not returnframe_locations:
        return batches
    else:
        return batches, frame_locations


def split_square_texture(texture, num_frames, max_frames, _smol_resolution, ebsynth=False):
    texture_height, texture_width = texture.shape[:2]
    texture_aspect_ratio = float(texture_width) / float(texture_height)

    frames_per_row = int(math.ceil(math.sqrt(max_frames)))
    frame_height = int(texture_height / frames_per_row)
    frame_width = int(texture_width / frames_per_row)

    _smol_frame_height = _smol_resolution
    _smol_frame_width = int(_smol_frame_height * texture_aspect_ratio)

    if ebsynth == False:
        _smol_frame_resized_width, _smol_frame_resized_height = utilityb.resize_to_nearest_multiple_of_8(
            _smol_frame_width, _smol_frame_height)
    else:
        _smol_frame_resized_width, _smol_frame_resized_height = _smol_frame_width, _smol_frame_height
        # _smol_frame_resized_width, _smol_frame_resized_height = _smol_frame_width, _smol_frame_height
    frames = []

    for i in range(num_frames):
        row, col = i // frames_per_row, i % frames_per_row
        frame = texture[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width]

        if not frame.size == 0:
            resized_frame = cv2.resize(frame, (_smol_frame_resized_width, _smol_frame_resized_height),
                                       interpolation=cv2.INTER_AREA)
            frames.append(resized_frame)
        else:
            tqdm.write("Frame size 0")
            frames.append(np.zeros((_smol_frame_resized_width, _smol_frame_resized_height, 3), dtype=np.uint8))

    return frames


def save_square_texture(texture, file_path):
    # Check if the input has the correct data type and convert if necessary
    if texture.dtype != np.uint8:
        texture = (texture * 255).astype(np.uint8)

    # Check if the input has the intended shape (3 channels for an RGB image)
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError("Invalid texture shape. Expected a 3-channel RGB image.")

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(texture)

    # Save the image to the specified file path
    tqdm.write(f'Saved to {file_path} at size {image.size}')
    image.save(file_path, format="PNG")


def convert_video_to_bytes(input_file):
    # Read the uploaded video file
    tqdm.write(f"Reading video file... {input_file}")
    with open(input_file, "rb") as f:
        video_bytes = f.read()

    # Return the processed video bytes (or any other output you want)
    return video_bytes


def generate_square_from_video(video_path, fps, batch_size, resolution, size_size):
    video_data = convert_video_to_bytes(video_path)
    frames_limit = (size_size * size_size) * batch_size
    frames = utilityb.extract_frames_movpy(video_data, fps, frames_limit)
    # print(len(frames))  #debug
    number_of_batches = size_size * size_size
    batches = split_into_batches(frames, batch_size, number_of_batches)
    tqdm.write(f"Number of batches: {len(batches)}")
    first_frames = [batch[0] for batch in batches]
    square_texture = create_square_texture(first_frames, resolution, side_length=size_size)
    # save_square_texture(square_texture, "./result/original.png")

    return square_texture


def generate_squares_to_folder(video_path, fps, batch_size, resolution, size_size, max_key_frames, output_folder,
                               border,
                               ebsynth_mode, max_frames_to_save, *args, pbar=None, gr_progress=None,
                               gr_progress_total_step=None, gr_progress_proceed_step=0):
    fps = int(fps)
    if not ebsynth_mode:
        if border >= (batch_size * size_size * size_size) / 2:
            gradio.Warning("Too many border frames, reduce border or increase batch size")
            raise Exception("Too many border frames, reduce border or increase batch size",
                            (batch_size * size_size * size_size) / 2, batch_size, size_size)

    # create folder
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

    # create folder for frames and keys if ebsynth mode is on
    if ebsynth_mode:
        if not os.path.exists(frames_loc):
            os.makedirs(frames_loc)
        if not os.path.exists(keys_loc):
            os.makedirs(keys_loc)

    # get video data
    video_data = convert_video_to_bytes(video_path)
    per_batch_limit = ((size_size * size_size) * batch_size) + border
    # if ebsynth_mode == False:
    #    per_batch_limit = per_batch_limit + border
    frames = utilityb.extract_frames_movpy(video_data, fps, max_key_frames, perform_interpolation=False)
    # print(per_batch_limit, size_size, batch_size, border)  # debug

    # split frames into batches
    big_batches = split_frames_into_big_batches(frames, per_batch_limit, border, ebsynth=ebsynth_mode)
    square_textures = []
    height = 0
    width = 0

    # if gr_progress_total_step is None, set it to len(big_batches)
    gr_progress_total_step = len(big_batches) if gr_progress_total_step is None else gr_progress_total_step

    with tqdm(total=len(big_batches), desc="Saving keyframes", position=2) as pbar1:
        for i in range(len(big_batches)):
            tqdm.write(f"Working on batch {i}")
            batches = split_into_batches(big_batches[i], batch_size, size_size * size_size)
            # print("Number of batches:", len(batches))  # debug
            if not ebsynth_mode:
                keyframes = [batch[0] for batch in batches]
            else:
                keyframes = [batch[int(len(batch) / 2)] for batch in batches]
            # for batch in batches:
            #   print (f"framenum = {int(len(batch)/2)} out of batch length {len(batch)} and size {len(frames)}")
            square_texture = create_square_texture(keyframes, resolution, side_length=size_size)
            save_square_texture(square_texture, os.path.join(input_folder_loc, f"input{i}.png"))
            square_textures.append(square_texture)
            height = square_texture.shape[0]
            width = square_texture.shape[1]

            # update progress bar
            pbar1.update(1)
            pbar is not None and pbar.update(1)
            gr_progress is not None and gr_progress((gr_progress_proceed_step + i) / gr_progress_total_step,
                                                    "Saving keyframes...")

    batch_settings_loc = os.path.join(output_folder, "batch_settings.txt")
    with open(batch_settings_loc, "w") as f:
        f.write(str(fps) + "\n")
        f.write(str(size_size) + "\n")
        f.write(str(batch_size) + "\n")
        f.write(str(video_path) + "\n")
        f.write(str(max_frames_to_save) + "\n")
        f.write(str(border) + "\n")
    # return list of urls

    return square_textures


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
                blended_image = cv2.addWeighted(current_batch[len(current_batch) - border + j], 1 - alpha,
                                                next_batch[j], alpha, 0)
                merged_batches.append(blended_image)
            except IndexError:
                tqdm.write("Merge failed")

    # Add remaining images from the last batch
    merged_batches.extend(image_batches[-1][border:])

    return merged_batches


def process_video_batch(video_path_old, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,
                        square_textures, max_frames, output_folder, border):
    video_path = os.path.join(output_folder, "input_video.mp4")
    per_batch_limmit = (((per_side * per_side) * batch_size)) + border
    video_data = convert_video_to_bytes(video_path)
    frames = utilityb.extract_frames_movpy(video_data, fps, max_frames)
    print(f"splitting into batches with per_batch_limmit = {per_batch_limmit} and border {border}")
    bigbatches = split_frames_into_big_batches(frames, per_batch_limmit, border, ebsynth=False)
    bigprocessedbatches = []
    for i, batch in enumerate(bigbatches):
        if i < len(square_textures):
            new_batch = process_video(batch, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,
                                      square_textures[i])
            bigprocessedbatches.append(new_batch)
            for a, image in enumerate(new_batch):
                Image.fromarray(image).save(os.path.join(output_folder, f"result/output{a + (len(new_batch) * i)}.png"))

    just_frame_groups = []
    print(f"bigprocessedbatches len = {len(bigprocessedbatches)}")
    for i in range(len(bigprocessedbatches)):
        newgroup = []
        for b in range(len(bigprocessedbatches[i])):
            newgroup.append(bigprocessedbatches[i][b])
        just_frame_groups.append(newgroup)

    combined = merge_image_batches(just_frame_groups, border)

    save_loc = os.path.join(output_folder, "blended.mp4")
    generated_vid = utilityb.pil_images_to_video(combined, save_loc, fps)
    return generated_vid


def process_video_single(video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,
                         square_texture):
    extension_path = os.path.abspath(__file__)
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    output_folder = os.path.join(extension_dir, "result")
    extension_path = os.path.abspath(__file__)
    frames_limit = (per_side * per_side) * batch_size
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    extension_save_folder = os.path.join(extension_dir, "result")
    if not os.path.exists(extension_save_folder):
        os.makedirs(extension_save_folder)
    utilityb.delete_folder_contents(extension_save_folder)
    # rerun the generatesquarefromvideo function to get the unaltered square texture
    video_data = convert_video_to_bytes(video_path)
    frames = utilityb.extract_frames_movpy(video_data, fps, frames_limit)
    processed_frames = process_video(frames, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,
                                     square_texture)
    output_video_path = os.path.join(output_folder, "output.mp4")
    generated_video = utilityb.pil_images_to_video(processed_frames, output_video_path, fps)
    return generated_video


def process_video(frames, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution, square_texture):
    frame_count = 0
    print(len(frames))
    batches = split_into_batches(frames, batch_size, per_side * per_side)
    print("Number of batches:", len(batches))
    first_frames = [batch[0] for batch in batches]
    # actuallyprocessthevideo
    debug = False

    frame_count = 0
    global resolution
    print(len(frames))

    if debug is False:
        # result_texture = sdprocess.square_Image_request(encoded_square_texture, prompt, initial_denoise, resolution, seed)
        result_texture = square_texture
        # save_square_texture(encoded_returns, "./result/processed.png")

    else:
        f = open("./result/processed.png", "rb")
        bytes = f.read()
        result_texture = base64.b64encode(bytes).decode("utf-8")
        resolution_get = Image.open("./result/processed.png")
        resolution = resolution_get.height
        # this is stupid and inefficiant i dont care
    # its not really encoded anymore is it
    encoded_returns = result_texture
    # encoded_returns = cv2.cvtColor(utilityb.base64_to_texture(result_texture), cv2.COLOR_BGR2RGB)

    new_frames = split_square_texture(encoded_returns, len(first_frames), per_side * per_side, _smol_resolution, False)

    if check_edges:
        for i, image in enumerate(new_frames):
            image = utilityb.check_edges(image)
    #:( 

    # Save first frames
    # for idx, first_frame in enumerate(first_frames):
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

        processed_batch, all_flow_after = sdprocess.batch_sd_run(encoded_batch, encoded_new_frame, frame_count, seed,
                                                                 False, fillindenoise, edgedenoise, _smol_resolution,
                                                                 True, encoded_new_frame, False)
        print(
            f"number {i} processed batch length {len(processed_batch)} and batch length {len(batch)} and num batches {len(batches)}")
        if last_processed is not None:
            encoded_batch.insert(0, utilityb.texture_to_base64(batches[i - 1][-1]))
            processed_batch_from_before, all_flow_after = sdprocess.batch_sd_run(encoded_batch, last_processed,
                                                                                 frame_count, seed, True, fillindenoise,
                                                                                 edgedenoise, _smol_resolution, True,
                                                                                 last_processed, False)
            if not min(len(processed_batch_from_before), len(processed_batch)) > 1:
                output_pil_images.append(
                    cv2.cvtColor(utilityb.base64_to_texture(processed_batch[0]), cv2.COLOR_BGR2RGB))
                continue
            blended_frames = blend_batches(processed_batch_from_before, processed_batch, resolution=_smol_resolution)
            print(f"blended frames {len(blended_frames)}")
            for b, blended_frame in enumerate(blended_frames):
                # savepath = os.path.join(output_folder, f"frame_{frame_count + b}.png")
                # save_square_texture(blended_frame, savepath)
                output_pil_images.append(blended_frame)
        else:
            for b, frame in enumerate(processed_batch):
                # savepath = os.path.join(output_folder, f"frame_{frame_count + b}.png")
                # save_square_texture(cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB), savepath)
                output_pil_images.append(cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB))

        frame_count += len(batch)
        last_processed = processed_batch[-1]
    print(f"output pil images {len(output_pil_images)}")
    return output_pil_images


def image_folder_to_video(folder_path, output_file, fps=24):
    """
    Turns a folder of images into a video using MoviePy.

    :param folder_path: str, path to the folder containing the images
    :param output_file: str, path to the output video file (e.g., 'output.mp4')
    :param fps: int, frames per second (default: 24)
    """
    # Get a list of image file names
    image_files = [f for f in os.listdir(folder_path) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Sort image files by their names
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Create a list of full image file paths
    image_paths = [os.path.join(folder_path, image) for image in image_files]

    # Create a clip from the image sequence
    clip = ImageSequenceClip(image_paths, fps=fps)

    # Write the clip to a video file
    clip.write_videofile(output_file, codec='libx264')

    return output_file


def blend_batches(batch_before, current_batch, resolution, blend_start_ratio=0.9, blend_end_ratio=0.1):
    blended_frames = []
    num_frames = min(len(batch_before), len(current_batch))
    decoded_batch_before = [cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB) for frame in
                            batch_before]
    decoded_current_batch = [cv2.cvtColor(utilityb.base64_to_texture(frame), cv2.COLOR_BGR2RGB) for frame in
                             current_batch]

    # target_width, target_height = resolution, resolution
    height, width = decoded_batch_before[0].shape[:2]
    # Resize the images in decoded_batch_before and decoded_current_batch
    decoded_batch_before = [cv2.resize(img, (width, height)) for img in decoded_batch_before]
    decoded_current_batch = [cv2.resize(img, (width, height)) for img in decoded_current_batch]

    output_folder = "moretemp"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not num_frames > 1:
        return [current_batch[0]]
    for i in range(num_frames):
        alpha = blend_start_ratio - (i / (num_frames - 1)) * (blend_start_ratio - blend_end_ratio)
        blended_frame = cv2.addWeighted(decoded_batch_before[i], alpha, decoded_current_batch[i], 1 - alpha, 0)
        print(f"blended frame {i}")

        # Concatenate the two input frames and the blended frame horizontally
        concatenated_frame = cv2.hconcat([decoded_batch_before[i], decoded_current_batch[i], blended_frame])

        cv2.imwrite(os.path.join(output_folder, f"concatenated_{i}.png"), concatenated_frame)

        blended_frames.append(blended_frame)
    return blended_frames


def interpolate_frames(frame1, frame2, alpha):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flags=0)
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
            interpolated_frame = interpolated_frame.transpose(1, 0,
                                                              2)  # Transpose back the dimensions of interpolated_frame (WxHxC to HxWxC)
            new_frames.append(interpolated_frame)

    new_frames.append(frames[-1])

    new_clip = ImageSequenceClip(new_frames, fps=output_fps)
    new_clip.write_videofile(output_path)
    return output_path


def split_videos_into_smaller_videos(max_keys, video, fps, max_frames, target_path, border_number, scenecuts=False,
                                     *args, pbar=None, gr_progress=None, gr_progress_total_step=None,
                                     gr_progress_proceed_step=0):
    max_total_frames = int((max_keys / 20) * max_frames)
    split_frames, border_indices = divideFrames(video, max_frames, border_number)
    split_frames_trimmed, trimmed_borders = trim_images(split_frames, max_total_frames, border_indices)
    # print(f" trim_imagestransitions {border_indices}") # debug
    # print(max_total_frames, border_indices, trimmed_borders, border_number, max_frames)  # debug
    output_files = []
    tqdm.write(
        f"Frames_total_size = {len(split_frames_trimmed)}, frames batch size = {max_frames} array length = {len(split_frames_trimmed)}")

    # if gr_progress_total_step is None else gr_progress_total_step
    gr_progress_total_step = len(split_frames_trimmed) if gr_progress_total_step is None else gr_progress_total_step

    # Split the video into smaller videos
    with tqdm(total=len(split_frames_trimmed), desc="Splitting video", position=2) as pbar1:
        for i, frames in enumerate(split_frames_trimmed):
            tqdm.write(f"Splitting video {i}")
            new_folder_location = os.path.join(target_path, f"{i}")

            if not os.path.exists(new_folder_location):
                os.makedirs(new_folder_location)

            new_video_loc = os.path.join(new_folder_location, f"input_video.mp4")
            output_files.append(utilityb.pil_images_to_video(frames, new_video_loc, fps))

            # update progress bar
            pbar1.update(1)
            pbar is not None and pbar.update(1)
            gr_progress is not None and gr_progress((gr_progress_proceed_step + i) / gr_progress_total_step,
                                                    "Splitting video...")

    return output_files, trimmed_borders


def divideFrames(frame_groups, x, y):
    result = []
    transitions = []

    for index, group in enumerate(frame_groups):
        # print(f"frame_groups {len(group)}")  # debug
        start = 0
        while start < len(group):
            end = start + x
            new_group = group[start:end]

            if end + y <= len(group):
                overlap_group = group[end:end + y]

                # Concatenate the images from new_group and overlap_group
                if y > 0:
                    combined_group = np.concatenate((new_group, overlap_group), axis=0)
                else:
                    combined_group = new_group
                # print(f"overlap group size {len(overlap_group)}")  # debug
                transitions.append(len(result))

            else:
                combined_group = new_group

            result.append(combined_group)
            start += x

    return result, transitions


def trim_images(images_list_of_lists, max_images, border_indices):
    """
    Trims the given list of lists of image arrays so that the total number of image arrays is below the specified maximum.
    Removes whole image arrays from the end of the list of lists if the max_images doesn't include them.

    Parameters:
    images_list_of_lists (list): List of lists of NumPy image arrays
    max_images (int): Maximum number of image arrays allowed

    Returns:
    list: List of lists of trimmed image arrays
    """
    total_images = sum([len(img_list) for img_list in images_list_of_lists])

    while total_images > max_images:
        # print(f"total_images = {total_images}, max_images = {max_images}")    # debug
        last_list_idx = len(images_list_of_lists) - 1
        last_img_idx = len(images_list_of_lists[last_list_idx]) - 1

        if last_img_idx >= 0:
            total_images -= 1
            images_list_of_lists[last_list_idx] = images_list_of_lists[last_list_idx][:-1]

        if len(images_list_of_lists[last_list_idx]) == 0:
            images_list_of_lists.pop()
            if last_list_idx in border_indices:
                border_indices.pop()

    return images_list_of_lists, border_indices
