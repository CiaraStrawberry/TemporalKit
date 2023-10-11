from __future__ import annotations

import base64
import glob
import io
import os
import re
import shutil
from collections import namedtuple

import cv2
import gradio as gr
import numpy as np
import scenedetect
import scripts.Berry_Method as General_SD
import scripts.Ebsynth_Processing as ebsynth
import scripts.berry_utility as sd_utility
from PIL import Image
from tqdm.auto import tqdm

import modules.generation_parameters_copypaste as parameters_copypaste
from modules import shared, script_callbacks

diffuseimg = None
SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])
lastmadefilename = ""


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def preprocess_video(video, fps, batch_size, per_side, resolution, batch_run, max_frames, output_path, border_frames,
                     ebsynth_mode, split_video, split_based_on_cuts):
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

        # max frames only applies in batch mode
        # otherwise it limmits the number of *total frames*
        border_frames = border_frames * batch_size

        max_frames = (20 * batch_size) - border_frames
        max_total_frames = int((max_keys / 20) * max_frames)
        existing_frames = []

        if split_based_on_cuts == True:
            existing_frames = sd_utility.split_video_into_numpy_arrays(video, fps, False)
        else:
            data = General_SD.convert_video_to_bytes(video)
            existing_frames = [
                sd_utility.extract_frames_movpie(data, fps, max_frames=max_total_frames, perform_interpolation=False)]

        split_video_paths, transition_data = General_SD.split_videos_into_smaller_videos(max_keys, existing_frames, fps,
                                                                                         max_frames, output_path,
                                                                                         border_frames,
                                                                                         split_based_on_cuts)
        for index, individual_video in enumerate(split_video_paths):

            generated_textures = General_SD.generate_squares_to_folder(individual_video, fps=fps, batch_size=batch_size,
                                                                       resolution=resolution, size_size=per_side,
                                                                       max_frames=None,
                                                                       output_folder=os.path.dirname(individual_video),
                                                                       border=0, ebsynth_mode=ebsynth_mode,
                                                                       max_frames_to_save=max_frames)
            input_location = os.path.join(os.path.dirname(os.path.dirname(individual_video)), "input")
            for tex_index, texture in enumerate(generated_textures):
                individual_file_name = os.path.join(input_location, f"{index}and{tex_index}.png")
                General_SD.save_square_texture(texture, individual_file_name)
        transitiondatapath = os.path.join(output_path, "transition_data.txt")
        with open(transitiondatapath, "w") as f:
            f.write(str(transition_data) + "\n")
            f.write(str(border_frames) + "\n")
        main_video_path = os.path.join(output_path, "main_video.mp4")
        sd_utility.copy_video(video, main_video_path)
        return main_video_path

    new_video_loc = os.path.join(output_path, f"input_video.mp4")
    shutil.copyfile(video, new_video_loc)
    if ebsynth_mode == True:
        border = 0

        image = General_SD.generate_squares_to_folder(video, fps=fps, batch_size=batch_size, resolution=resolution,
                                                      size_size=per_side, max_frames=max_frames,
                                                      output_folder=output_path, border=border_frames,
                                                      ebsynth_mode=True, max_frames_to_save=max_frames)
        return image[0]

    if batch_run == False:
        image = General_SD.generate_square_from_video(video, fps=fps, batch_size=batch_size, resolution=resolution,
                                                      size_size=per_side)
        processed = numpy_array_to_temp_url(image)
    else:
        image = General_SD.generate_squares_to_folder(video, fps=fps, batch_size=batch_size, resolution=resolution,
                                                      size_size=per_side, max_frames=max_frames,
                                                      output_folder=output_path, border=border_frames,
                                                      ebsynth_mode=False, max_frames_to_save=max_frames)
        processed = image[0]
    return processed


def apply_image_to_video(image, video, fps, per_side, output_resolution, batch_size):
    return General_SD.process_video_single(video_path=video, fps=fps, per_side=per_side, batch_size=batch_size,
                                           fillindenoise=0, edgedenoise=0, _smol_resolution=output_resolution,
                                           square_texture=image)


def apply_image_to_vide_batch(input_folder, video, fps, per_side, output_resolution, batch_size, max_frames,
                              border_frames):
    input_images_folder = os.path.join(input_folder, "output")
    images = read_images_folder(input_images_folder)
    print(len(images))
    return General_SD.process_video_batch(video_path_old=video, fps=fps, per_side=per_side, batch_size=batch_size,
                                          fillindenoise=0, edgedenoise=0, _smol_resolution=output_resolution,
                                          square_textures=images, max_frames=max_frames, output_folder=input_folder,
                                          border=border_frames)


def post_process_ebsynth(input_folder, video, fps, per_side, output_resolution, batch_size, max_frames, border_frames,
                         progress=gr.Progress()):
    """
    Post processes the ebsynth output into a video

    :param input_folder: The folder containing the ebsynth output
    :param video: The video that was used to generate the ebsynth output
    :param fps: The fps of the video
    :param per_side: The number of squares per side
    :param output_resolution: The resolution of the output video
    :param batch_size: The number of frames per keyframe
    :param max_frames: The maximum number of frames to process
    :param border_frames: The number of border frames
    :param progress: The progress bar to update

    :return: The path to the generated video
    """

    input_images_folder = os.path.join(input_folder, "output")
    split_mode = os.path.join(input_folder, "batch_settings.txt")

    # If the keys folder exists, use the sort_into_folders function
    if os.path.exists(split_mode):
        images = read_images_folder(input_images_folder)
        ebsynth.sort_into_folders(video_path=video, fps=fps, per_side=per_side, batch_size=batch_size,
                                  _smol_resolution=output_resolution, square_textures=images,
                                  max_frames=max_frames, output_folder=input_folder, border=border_frames,
                                  progress=progress)
        gr.Info("Finished processing")
        return [os.path.join(input_folder, "keys", filename) for filename in os.listdir(os.path.join(input_folder, "keys"))]
    else:
        img_folder = os.path.join(input_folder, "output")
        # define a regular expression pattern to match directory names with one or more digits
        pattern = r'^\d+$'

        # get a list of all directories in the specified path
        all_dirs = os.listdir(input_folder)

        # use a list comprehension to filter the directories based on the pattern
        numeric_dirs = sorted([d for d in all_dirs if re.match(pattern, d)], key=lambda x: int(x))
        max_frames = max_frames + border_frames

        with tqdm(position=2, desc="Working dir", total=len(numeric_dirs) - 1) as pbar:
            for i, d in enumerate(numeric_dirs):
                # create a list to store the filenames of the images that match the directory name
                img_names = []
                folder_video = os.path.join(input_folder, d, "input_video.mp4")

                # batch settings in sub folder
                read_path = os.path.join(input_folder, d, "batch_settings.txt")
                with open(read_path, "r") as f:
                    fps = int(f.readline().strip())
                    per_side = int(f.readline().strip())
                    batch_size = int(f.readline().strip())
                    f.readline().strip()
                    max_frames = int(f.readline().strip())
                    border_frames = int(f.readline().strip())

                # loop through each image file in the image folder
                for img_file in os.listdir(img_folder):
                    # check if the image filename starts with the directory name followed by the word "and" and a sequence of one or more digits, then ends with '.png'
                    if re.match(f"^.*{d}and\d+.*\.png$", img_file):
                        img_names.append(img_file)
                tqdm.write(f"Post processing = {os.path.dirname(folder_video)}")
                square_textures = []

                # loop through each image file name
                for img_name in sorted(img_names, key=lambda x: int(re.search(r'and(\d+)', x).group(1))):
                    img = Image.open(os.path.join(input_images_folder, img_name))
                    # Convert image to NumPy array and append to images list
                    tqdm.write(f"Read output keyframe {os.path.join(input_images_folder, img_name)}")
                    square_textures.append(np.array(img))

                ebsynth.sort_into_folders(video_path=folder_video, fps=fps, per_side=per_side, batch_size=batch_size,
                                          _smol_resolution=output_resolution, square_textures=square_textures,
                                          max_frames=max_frames, output_folder=os.path.dirname(folder_video),
                                          border=border_frames, progress=progress, index_dir=i,
                                          total_dir=len(numeric_dirs))

                # update the progress bar
                pbar.update(1)

        gr.Info("Finished processing")
        return [os.path.join(input_folder, '0', "keys", filename) for filename in os.listdir(os.path.join(input_folder, '0', "keys"))]


def recombine_ebsynth(input_folder, fps, border_frames, batch):
    if os.path.exists(os.path.join(input_folder, "keys")):
        return ebsynth.crossfade_folder_of_folders(input_folder, fps=fps, return_generated_video_path=True)
    else:
        generated_videos = []
        pattern = r'^\d+$'

        # get a list of all directories in the specified path
        all_dirs = os.listdir(input_folder)

        # use a list comprehension to filter the directories based on the pattern
        numeric_dirs = sorted([d for d in all_dirs if re.match(pattern, d)], key=lambda x: int(x))

        for d in numeric_dirs:
            folder_loc = os.path.join(input_folder, d)
            # loop through each image file in the image folder
            new_video = ebsynth.crossfade_folder_of_folders(folder_loc, fps=fps)
            # print(f"generated new video at location {new_video}")
            generated_videos.append(new_video)

        overlap_data_path = os.path.join(input_folder, "transition_data.txt")
        with open(overlap_data_path, "r") as f:
            merge = str(f.readline().strip())

        overlap_indicies = []
        int_list = eval(merge)
        for num in int_list:
            overlap_indicies.append(int(num))

        output_video = sd_utility.crossfade_videos(video_paths=generated_videos, fps=fps,
                                                   overlap_indexes=overlap_indicies, num_overlap_frames=border_frames,
                                                   output_path=os.path.join(input_folder, "output.mp4"))
        return output_video
    return None


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    split = re.split(r'(\d+)', text)
    for i in range(len(split), 0, -1):
        if split[i - 1].isdigit():
            return int(split[i - 1])


def read_images_folder(folder_path):
    images = []
    filenames = os.listdir(folder_path)

    # Sort filenames based on the order of the numbers in their names
    filenames.sort(key=natural_keys)

    for filename in filenames:
        # Check if file is an image (assumes only image files are in the folder)
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            if re.match(r".*(input).*", filename):
                # Open image using Pillow library

                img = Image.open(os.path.join(folder_path, filename))
                # Convert image to NumPy array and append to images list
                images.append(np.array(img))
            else:
                print(f"[${filename}] File name must contain \"input\". Skip processing.")
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
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    extension_folder = os.path.join(extension_dir, "squares")
    if not os.path.exists(extension_folder):
        os.makedirs(extension_folder)
        # create a path for the temporary file
    file_path = os.path.join(extension_folder, filename)

    # convert the array to an image using PIL
    img = Image.fromarray(img_array)

    # save the image to the temporary file as PNG
    img.save(file_path, format='PNG')

    # create a URL for the temporary file
    # url = 'file://' + file_path

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


def read_video_setting(video_path):
    """
    Reads the fps and keyframes from the provided video

    :param video_path: The path to the video
    :return: The fps and keyframes
    """
    if video_path == "":
        return 24, 4

    video = scenedetect.VideoManager(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    keyframes = round(fps / 6)
    return fps, keyframes


def update_image():
    global diffuseimg
    extension_path = os.path.abspath(__file__)
    # get the directory name of the extension
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    extension_folder = os.path.join(extension_dir, "squares")
    most_recent_image = get_most_recent_file(extension_folder)
    print(most_recent_image)
    pilImage = Image.open(most_recent_image)
    print("running")
    return most_recent_image


def update_settings():
    extension_path = os.path.abspath(__file__)
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    tempfile = os.path.join(extension_dir, "temp_file.txt")
    with open(tempfile, "r") as f:
        fps = int(f.readline().strip())
        sides = int(f.readline().strip())
        batch_size = int(f.readline().strip())
        video_path = f.readline().strip()
    return fps, sides, batch_size, video_path


def update_settings_from_file(folder_path):
    """
    Reads the settings from the batch_settings.txt or file in the provided folder path

    :param folder_path: The path to the folder containing the batch_settings.txt file
    :return: The fps, sides, batch_size, video_path, max_frames, border
    """
    read_path = os.path.join(folder_path, "batch_settings.txt")
    transition_data_path = os.path.join(folder_path, "transition_data.txt")

    # If the batch_settings.txt file doesn't exist, check if the transition_data.txt file exists
    if not (os.path.exists(read_path) or os.path.exists(transition_data_path)):
        gr.Warning('No batch settings found, Please make sure the path are correct.')
        return 10, 3, 5, None, 100, 1

    # If the transition_data.txt file exists, read the settings from there
    if not os.path.exists(read_path):
        print(f"Reading path at {transition_data_path}")
        read_path = os.path.join(folder_path, "0/batch_settings.txt")
        video_path = os.path.join(folder_path, "main_video.mp4")

        # Read the settings from the transition_data.txt file (unknown how this work)
        # with open(transition_data_path, "r") as b:
        #     merge = str(b.readline().strip())
        #     border = int(b.readline().strip())

        # Read the settings from the batch_settings.txt file
        with open(read_path, "r") as f:
            fps = int(f.readline().strip())
            sides = int(f.readline().strip())
            batch_size = int(f.readline().strip())
            f.readline().strip()
            max_frames = int(f.readline().strip())
            border = int(f.readline().strip())

    # Otherwise, read the settings from the batch_settings.txt file
    else:
        print(f"Reading path at {read_path}")
        with open(read_path, "r") as f:
            fps = int(f.readline().strip())
            sides = int(f.readline().strip())
            batch_size = int(f.readline().strip())
            f.readline().strip()
            video_path = os.path.join(folder_path, "input_video.mp4")
            max_frames = int(f.readline().strip())
            border = int(f.readline().strip())

    return fps, sides, batch_size, video_path, max_frames, border


def save_settings(fps, sides, batch_size, video):
    extension_path = os.path.abspath(__file__)
    extension_dir = os.path.dirname(os.path.dirname(extension_path))
    tempfile = os.path.join(extension_dir, "temp_file.txt")
    with open(tempfile, "w") as f:
        f.write(str(fps) + "\n")
        f.write(str(sides) + "\n")
        f.write(str(batch_size) + "\n")
        f.write(str(video) + "\n")


def create_video_Processing_Tab():
    with gr.Column(visible=True, elem_id="Temporal_Kit") as main_panel:
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.TabItem(elem_id="output_TemporalKit", label="Input"):
                    with gr.Column():
                        video = gr.Video(label="Input Video", elem_id="input_video", type="filepath")
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                sides = gr.Number(value=1, label="Sides", precision=0, interactive=True)
                                resolution = gr.Number(value=512, label="Height Resolution", precision=1,
                                                       interactive=True)
                                batch_size = gr.Number(value=4, label="Frames per keyframe", precision=1,
                                                       interactive=True)
                                fps = gr.Number(value=24, precision=1, label="fps", interactive=True)
                            with gr.Row():
                                ebsynth_mode = gr.Checkbox(label="EBSynth Mode", value=True)
                            with gr.Column():
                                savesettings = gr.Button("Save Settings")

                        with gr.Row():
                            batch_folder = gr.Textbox(label="Target Folder",
                                                      placeholder="This is ignored if neither batch run or ebsynth are checked")

                        with gr.Row():
                            with gr.Accordion("Batch Settings", open=False):
                                with gr.Row():
                                    batch_checkbox = gr.Checkbox(label="Batch Run", value=False)
                                    max_keyframes = gr.Number(value=-1, label="Max key frames", precision=1,
                                                              interactive=True, placeholder="for all frames")
                                    border_frames = gr.Number(value=0, label="Border Key Frames", precision=1,
                                                              interactive=True, placeholder="border frames")
                        with gr.Row():
                            with gr.Accordion("EBSynth Settings", open=False):
                                with gr.Row():
                                    split_video = gr.Checkbox(label="Split Video", value=False)
                                    split_based_on_cuts = gr.Checkbox(label="Split based on cuts (as well)",
                                                                      value=False)
                                    # interpolate = gr.Checkbox(label="Interpolate(high memory)", value=False)

            with gr.Tabs(elemn_id="TemporalKit_gallery_container"):
                with gr.TabItem(elem_id="output_TemporalKit", label="Output"):
                    with gr.Row():
                        result_image = gr.Image(type='pil', interactive=False, label="Output Image", )
                    with gr.Row():
                        runbutton = gr.Button("Run", variant="primary")
                    with gr.Row():
                        send_to_buttons = parameters_copypaste.create_buttons(["img2img"])

                try:
                    parameters_copypaste.bind_buttons(send_to_buttons, result_image, [""])
                except Exception as e:
                    print(e)

    video.change(
        fn=read_video_setting,
        inputs=[video],
        outputs=[fps, batch_size])
    savesettings.click(
        fn=save_settings,
        inputs=[fps, sides, batch_size, video]
    )
    parameters_copypaste.add_paste_fields("TemporalKit", result_image, None)
    runbutton.click(fn=preprocess_video,
                    inputs=[video, fps, batch_size, sides, resolution, batch_checkbox, max_keyframes, batch_folder,
                            border_frames, ebsynth_mode, split_video, split_based_on_cuts],
                    outputs=result_image)


def show_textbox(option):
    if option == True:
        return gr.inputs.Textbox(lines=2, placeholder="Enter your text here")
    else:
        return False


def create_diffusing_tab():
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
                                fps = gr.Number(label="FPS", value=10, precision=1)
                                per_side = gr.Number(label="per side", value=3, precision=1)
                                output_resolution_single = gr.Number(label="output height resolution", value=1024,
                                                                     precision=1)
                                batch_size_diffuse = gr.Number(label="batch size", value=10, precision=1)
                            with gr.Row():
                                runButton = gr.Button("run", elem_id="run_button")

            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Output"):
                        with gr.Column():
                            # newbutton = gr.Button("update", elem_id="update_button")
                            outputfile = gr.Video()

        read_last_image.click(
            fn=update_image,
            outputs=input_image
        )
        read_last_settings.click(
            fn=update_settings,
            outputs=[fps, per_side, batch_size_diffuse, input_video]
        )
        runButton.click(
            fn=apply_image_to_video,
            inputs=[input_image, input_video, fps, per_side, output_resolution_single, batch_size_diffuse],
            outputs=outputfile
        )


def create_batch_tab():
    with gr.Column(visible=True, elem_id="batch_process") as second_panel:
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Generate Batch"):
                        with gr.Column():
                            with gr.Row():
                                input_folder = gr.Textbox(label="Input Folder",
                                                          placeholder="the whole folder, generated before, not just the output folder")
                                input_video = gr.Video(label="Input Video", elem_id="input_videopage2")
                            with gr.Row():
                                read_last_settings = gr.Button("read_last_settings", elem_id="read_last_settings")
                            with gr.Row():
                                fps = gr.Number(label="FPS", value=10, precision=1)
                                per_side = gr.Number(label="per side", value=3, precision=1)
                                output_resolution_batch = gr.Number(label="output resolution", value=1024, precision=1)
                                batch_size = gr.Number(label="batch size", value=5, precision=1)
                                max_frames = gr.Number(label="max frames", value=100, precision=1)
                                border_frames = gr.Number(label="border frames", value=1, precision=1)
                            with gr.Row():
                                runButton = gr.Button("run", elem_id="run_button")

            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Output"):
                        with gr.Column():
                            # newbutton = gr.Button("update", elem_id="update_button")
                            outputfile = gr.Video()

        read_last_settings.click(
            fn=update_settings_from_file,
            inputs=[input_folder],
            outputs=[fps, per_side, batch_size, input_video, max_frames, border_frames]
        )
        runButton.click(
            fn=apply_image_to_vide_batch,
            inputs=[input_folder, input_video, fps, per_side, output_resolution_batch, batch_size, max_frames,
                    border_frames],
            outputs=outputfile
        )


def create_ebsynth_tab():
    with gr.Column(visible=True, elem_id="batch_process") as second_panel:
        with gr.Row():
            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Generate"):
                        with gr.Column():
                            with gr.Row(variant="panel"):
                                with gr.Row():
                                    input_folder = gr.Textbox(label="Input Folder (Target Folder)",
                                                              placeholder="The whole folder, generated before, not just "
                                                                          "the output folder")
                                with gr.Row():
                                    read_last_settings_synth = gr.Button("Read Settings", elem_id="read_last_settings")
                            with gr.Row():
                                input_video = gr.Video(label="Input Video", elem_id="input_videopage2", type="filepath",
                                                       interactive=False)
                            with gr.Row(variant="panel"):
                                fps = gr.Number(label="Video FPS", value=10, precision=1)
                                per_side = gr.Number(label="Side", value=3, precision=1)
                                output_resolution_batch = gr.Number(label="Output images resolution", value=1024,
                                                                    precision=1)
                                batch_size = gr.Number(label="Frames per keyframe", value=5, precision=1)
                                max_frames = gr.Number(label="Max frames", value=0, precision=1)
                                border_frames = gr.Number(value=1, label="Border Frames", precision=1, interactive=True,
                                                          placeholder="border frames")
                                with gr.Row():
                                    create_ebs = gr.Checkbox(label="Create ebsynth file (.ebs)", value=True)
                            with gr.Row():
                                run_button = gr.Button("Prepare ebsynth", elem_id="run_button", variant="primary")
                                recombine_button = gr.Button("Recombine ebsynth", elem_id="recombine_button",
                                                             variant="primary")

            with gr.Tabs(elem_id="mode_TemporalKit"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Output"):
                        with gr.Column():
                            # newbutton = gr.Button("update", elem_id="update_button")
                            output_file = gr.File(interactive=False)

        read_last_settings_synth.click(
            fn=update_settings_from_file,
            inputs=[input_folder],
            outputs=[fps, per_side, batch_size, input_video, max_frames, border_frames]
        )
        input_folder.input(
            fn=update_settings_from_file,
            inputs=[input_folder],
            outputs=[fps, per_side, batch_size, input_video, max_frames, border_frames]
        )
        run_button.click(
            fn=post_process_ebsynth,
            inputs=[input_folder, input_video, fps, per_side, output_resolution_batch, batch_size, max_frames,
                    border_frames],
            outputs=output_file
        )
        recombine_button.click(
            fn=recombine_ebsynth,
            inputs=[input_folder, fps, border_frames, batch_size],
            outputs=output_file
        )


tabs_list = ["TemporalKit"]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as temporalkit:
        with gr.Tabs(elem_id="TemporalKit-Tab") as tabs:
            with gr.Tab(label="Pre-Processing"):
                with gr.Blocks(analytics_enabled=False):
                    create_video_Processing_Tab()
            with gr.Tab(label="Temporal-Warp", elem_id="processbutton"):
                with gr.Blocks(analytics_enabled=False):
                    create_diffusing_tab()
            with gr.Tab(label="Batch-Warp", elem_id="batch-button"):
                with gr.Blocks(analytics_enabled=False):
                    create_batch_tab()
            with gr.Tab(label="Ebsynth-Process", elem_id="Ebsynth-Process"):
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
    shared.opts.add_option("def_img_cfg",
                           shared.OptionInfo("1.5", "Default Image CFG", section=('ip2p', "Instruct-pix2pix")))


from base64 import b64decode, b64encode
from io import BytesIO


def img_to_b64(image: Image.Image):
    buf = BytesIO()
    image.save(buf, format="png")
    return b64encode(buf.getvalue()).decode("utf-8")


def b64_to_img(enc: str):
    if enc.startswith('data:image'):
        enc = enc[enc.find(',') + 1:]
    return Image.open(BytesIO(b64decode(enc)))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
