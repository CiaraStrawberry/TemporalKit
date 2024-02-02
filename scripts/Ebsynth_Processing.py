import os
import re

import cv2
import gradio as gr
# om nom nom nom
import numpy as np
import scripts.Berry_Method as bmethod
import scripts.berry_utility as butility
from PIL import Image
from tqdm.auto import tqdm

import extensions.TemporalKit.scripts.berry_utility
from extensions.TemporalKit.scripts.ebsynthfile import EBSynthProject


def sort_into_folders(video_path, fps, per_side, batch_size, _smol_resolution, square_textures, max_frames,
                      output_folder, border=0, *args, progress=gr.Progress(), index_dir=0, total_dir=1, ebs_file=False):
    """
    Sort the frames of a video into folders for use with EBSynth.

    :param video_path: Path to the video to process.
    :param fps: Frames per second of the video.
    :param per_side: Number of frames per side of the square texture.
    :param batch_size: Number of frames per batch.
    :param _smol_resolution: Resolution of the output frames.
    :param square_textures: List of square textures to use.
    :param max_frames: Maximum number of frames to process.
    :param output_folder: Path to the folder to save the output to.
    :param border: Border size to use.
    :param progress: Gradio progress bar.
    :param index_dir: Index of the current directory.
    :param total_dir: Total number of directories.
    """

    per_batch_limit = ((per_side * per_side) * batch_size) + border
    ebs_project = EBSynthProject("frames/frames[#####].png", "keys/keys[#####].png", "", False)
    #  original_frames_directory = os.path.join(output_folder, "original_frames")
    #  if os.path.exists(original_frames_directory):
    #      for filename in os.listdir(original_frames_directory):
    #          frames.append(cv2.imread(os.path.join(original_frames_directory, filename), cv2.COLOR_BGR2RGB))
    #  else:

    video_data = bmethod.convert_video_to_bytes(video_path)
    frames = butility.extract_frames_movpy(video_data, fps, max_frames)
    input_folder = os.path.join(output_folder, "input")  # input folder for ebsynth
    total_frames = len(frames)
    tqdm.write(f"Full frames num = {total_frames}")

    # progress bar
    progress_total = total_frames + int(np.ceil(total_frames / batch_size))  # tqdm progress bar
    # total progress for all directories for gradio progress bar
    gr_progress_total = (total_frames + int(np.ceil(total_frames / batch_size))) * total_dir
    # already done in previous directories for gradio progress bar
    gr_progress_already_done = index_dir * (total_frames + int(np.ceil(total_frames / batch_size)))
    progress(gr_progress_already_done / gr_progress_total, "Loading...")  # update gradio progress bar

    # create output, keys folders if they don't exist
    output_frames_folder = os.path.join(output_folder, "frames")
    if not os.path.exists(output_frames_folder):
        os.makedirs(output_frames_folder)
    output_keys_folder = os.path.join(output_folder, "keys")
    if not os.path.exists(output_keys_folder):
        os.makedirs(output_keys_folder)

    # get original frame size
    filenames = os.listdir(input_folder)
    img = Image.open(os.path.join(input_folder, filenames[0]))
    original_width, original_height = img.size
    height, width = frames[0].shape[:2]

    # calculate aspect ratio
    texture_aspect_ratio = float(width) / float(height)

    # calculate new frame size
    _smol_frame_height = _smol_resolution
    _smol_frame_width = int(_smol_frame_height * texture_aspect_ratio)
    tqdm.write(f"Saving size = {_smol_frame_width}x{_smol_frame_height}")

    with tqdm(total=progress_total, position=1, desc="Total") as pbar1:

        # save original frames
        with tqdm(total=total_frames, position=0, desc="Saving frames") as pbar2:
            for i, frame in enumerate(frames):
                frame_to_save = cv2.resize(frame, (_smol_frame_width, _smol_frame_height),
                                           interpolation=cv2.INTER_LINEAR)
                bmethod.save_square_texture(frame_to_save,
                                            os.path.join(output_frames_folder, "frames{:05d}.png".format(i)))

                # update progress bars
                pbar2.update(1)
                pbar1.update(1)
                progress((i + gr_progress_already_done) / gr_progress_total, "Saving frames...")
        # original_frame_height, original_frame_width = frames[0].shape[:2]

        # split frames into batches
        big_batches, frame_locs = bmethod.split_frames_into_big_batches(frames, per_batch_limit, border, ebsynth=True,
                                                                        returnframe_locations=True)
        big_processed_batches = []
        last_frame_end = 0
        # print(len(square_textures))  # debug

        # save keyframes
        with tqdm(total=int(np.ceil(total_frames / batch_size)), position=0, desc="Saving keyframes") as pbar2:
            for a, big_batch in enumerate(big_batches):
                batches = bmethod.split_into_batches(big_batch, batch_size, per_side * per_side)
                keyframes = [batch[int(len(batch) / 2)] for batch in batches]

                # if there are more square textures than keyframes, use the last square texture for the remaining keyframes
                if a < len(square_textures):
                    resized_square_texture = cv2.resize(square_textures[a], (original_width, original_height),
                                                        interpolation=cv2.INTER_LINEAR)
                    new_frames = bmethod.split_square_texture(resized_square_texture, len(keyframes),
                                                              per_side * per_side,
                                                              _smol_resolution, True)
                    new_frame_start, new_frame_end = frame_locs[a]

                    for b in range(len(new_frames)):
                        # print(new_frame_start)  # debug
                        inner_start = last_frame_end
                        inner_end = inner_start + len(batches[b])
                        last_frame_end = inner_end
                        frame_position = inner_start + int((inner_end - inner_start) / 2)
                        tqdm.write(f"Saving at frame {frame_position}")
                        frame_to_save = cv2.resize(new_frames[b], (_smol_frame_width, _smol_frame_height),
                                                   interpolation=cv2.INTER_LINEAR)
                        bmethod.save_square_texture(
                            frame_to_save, os.path.join(output_keys_folder, "keys{:05d}.png".format(frame_position)))

                    # ebsynth file
                    ebs_project.AddKeyFrame(True, True, max(0, frame_position - len(batches[b])), frame_position,
                                            min(len(frames)-1, frame_position + len(batches[b])),
                                            "out_{:05d}/[#####].png".format(frame_position))

                    # update progress bars
                    pbar2.update(1)
                    pbar1.update(1)
                    progress((total_frames + (a + 1) + gr_progress_already_done) / gr_progress_total,
                             "Saving keyframes...")

            # save ebsynth file
            ebs_project.WriteToFile(os.path.join(output_folder, "keys.ebs"))
            ebs_project.keyFrames.clear()

        # TODO: unknown what this does (by cocomine)
        just_frame_groups = []
        for i in range(len(big_processed_batches)):
            newgroup = []
            for b in range(len(big_processed_batches[i])):
                newgroup.append(big_processed_batches[i][b])
            just_frame_groups.append(newgroup)

        return


def recombine(video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution, square_textures,
              max_frames, output_folder, border):
    just_frame_groups = []
    per_batch_limmit = (((per_side * per_side) * batch_size)) - border
    video_data = bmethod.convert_video_to_bytes(video_path)
    frames = bmethod.extract_frames_movpy(video_data, fps, max_frames)
    bigbatches, frameLocs = bmethod.split_frames_into_big_batches(frames, per_batch_limmit, border,
                                                                  returnframe_locations=True)
    bigprocessedbatches = []
    for i in range(len(bigprocessedbatches)):
        newgroup = []
        for b in range(len(bigprocessedbatches[i])):
            newgroup.append(bigprocessedbatches[i][b])
        just_frame_groups.append(newgroup)

    combined = bmethod.merge_image_batches(just_frame_groups, border)

    save_loc = os.path.join(output_folder, "non_blended.mp4")
    generated_vid = extensions.TemporalKit.scripts.berry_utility.pil_images_to_video(combined, save_loc, fps)


def crossfade_folder_of_folders(output_folder, fps, return_generated_video_path=False):
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

    with tqdm(total=len(dirs) - 1, position=0, desc="Crossfading") as pbar:
        for i in range(len(dirs) - 1):
            current_dir = os.path.join(root_folder, dirs[i])
            next_dir = os.path.join(root_folder, dirs[i + 1])

            images_current = sorted(os.listdir(current_dir))
            images_next = sorted(os.listdir(next_dir))

            startnum = get_num_at_index(current_dir, 0)
            bigkeynum = allkeynums[i]
            keynum = bigkeynum - startnum
            tqdm.write(f"recombining directory {dirs[i]} and {dirs[i + 1]}, len {keynum}")

            for j in range(keynum, len(images_current) - 1):
                alpha = (j - keynum) / (len(images_current) - keynum)
                image1_path = os.path.join(current_dir, images_current[j])
                next_image_index = j - keynum if j - keynum < len(images_next) else len(images_next) - 1
                image2_path = os.path.join(next_dir, images_next[next_image_index])

                image1 = Image.open(image1_path)
                image2 = Image.open(image2_path)

                blended_image = butility.crossfade_images(image1, image2, alpha)
                output_images.append(np.array(blended_image))
                # blended_image.save(os.path.join(output_folder, f"{dirs[i]}_{dirs[i+1]}_crossfade_{j:04}.png"))
            pbar.update(1)

    final_dir = os.path.join(root_folder, dirs[-1])
    final_dir_images = sorted(os.listdir(final_dir))

    # Find the index of the image with the last keyframe number in its name
    last_keyframe_number = allkeynums[-1]
    last_keyframe_index = None
    for index, image_name in enumerate(final_dir_images):
        number_in_name = int(''.join(filter(str.isdigit, image_name)))
        if number_in_name == last_keyframe_number:
            last_keyframe_index = index
            break

    if last_keyframe_index is not None:
        print(f"going from dir {last_keyframe_number} to end at {len(final_dir_images)}")

        # Iterate from the last keyframe number to the end
        for c in range(last_keyframe_index, len(final_dir_images)):
            image1_path = os.path.join(final_dir, final_dir_images[c])
            image1 = Image.open(image1_path)
            output_images.append(np.array(image1))
    else:
        print("Last keyframe not found in the final directory")

    print(f"outputting {len(output_images)} images")
    output_save_location = os.path.join(output_folder, "crossfade.mp4")
    generated_vid = extensions.TemporalKit.scripts.berry_utility.pil_images_to_video(output_images,
                                                                                     output_save_location, fps)

    if return_generated_video_path == True:
        return generated_vid
    else:
        return output_images


def getkeynums(folder_path):
    filenames = os.listdir(folder_path)

    # Filter filenames to keep only the ones starting with "keys" and ending with ".png"
    keys_filenames = [f for f in filenames if f.startswith("keys") and f.endswith(".png")]

    # Sort the filtered filenames
    sorted_keys_filenames = sorted(keys_filenames, key=lambda f: int(re.search(r'(\d+)', f).group(0)))

    # Extract the numbers from the sorted filenames
    return [int(re.search(r'(\d+)', f).group(0)) for f in sorted_keys_filenames]


def get_num_at_index(folder_path, index):
    """Get the starting number of the output images in a folder."""
    filenames = os.listdir(folder_path)

    # Filter filenames to keep only the ones starting with "keys" and ending with ".png"
    # keys_filenames = [f for f in filenames if f.startswith("keys") and f.endswith(".png")]

    # Sort the filtered filenames
    sorted_keys_filenames = sorted(filenames, key=lambda f: int(re.search(r'(\d+)', f).group(0)))

    # Extract the numbers from the sorted filenames
    numbers = [int(re.search(r'(\d+)', f).group(0)) for f in sorted_keys_filenames]
    return numbers[index]
