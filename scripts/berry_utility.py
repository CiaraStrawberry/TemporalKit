import os
import glob
from moviepy.editor import *
import tempfile
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
import copy
import shutil
import subprocess
import scenedetect

window_size = 5 

intensity_window = deque(maxlen=window_size)

def calculate_intensity(flow_map,frame_count):
    global min_intensity, max_intensity
    intensity_map = np.sqrt(np.sum(flow_map**2, axis=2))
    
    intensity_window.append(intensity_map)
    
    min_intensity = min(im.min() for im in intensity_window)
    max_intensity = max(im.max() for im in intensity_window)

    normalized_intensity_map  = (intensity_map - min_intensity) / (max_intensity - min_intensity)
    
    intensity_image = Image.fromarray((normalized_intensity_map * 255).astype(np.uint8))
    intensity_image.save(f'intensitymaps/intensity_map_{frame_count:04d}.png')
    
    return normalized_intensity_map 

def scale_mask_intensity(mask, intensity):
    scaled_mask = np.clip(mask * intensity, 0, 255).astype(np.uint8)
    return scaled_mask

def mask_to_grayscale(mask):
    grayscale_mask = 0.2989 * mask[:, :, 0] + 0.5870 * mask[:, :, 1] + 0.1140 * mask[:, :, 2]
    return grayscale_mask



def replace_masked_area(flow,index, base_image_path, mask_base64_str, replacement_image_path, threshold=128):

    #intensity = 0.2
    
    #intensity_map = calculate_intensity(flow)
    
    
    # Load images
    base_image = Image.open(base_image_path).convert("RGBA")
    #mask_image = Image.open(io.BytesIO(base64.b64decode(mask_base64_str))).convert("L")
    mask_image = mask_base64_str # its not base64 encoded anymore
    #replacement_tex_unmodified =  base64_to_texture(replacement_image_path)
    #replacement_image = Image.fromarray(np.uint8(replacement_tex_unmodified)).convert("RGBA")
    replacement_image = Image.open(replacement_image_path).convert("RGBA")
    print(mask_image)
    print(mask_image.size)

    # Resize the mask image and replacement image to match the base image size
    base_width, base_height = base_image.size
    alpha_mask = np.array(mask_image)

    mask_image = alpha_mask.resize((base_width, base_height))
    replacement_image = replacement_image.resize((base_width, base_height))

    alpha_mask_intensity = (alpha_mask).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_mask_intensity)
    
    blended_image = Image.composite(replacement_image, base_image, alpha_image)
  #  print(alpha_mask_intensity.size)
  #  print (f"image{intensity_map.size}")

    output_image_path = os.path.join("./debug3/", f"output_image_{index}.png")
    # Save the output image
    os.makedirs("./debug3/", exist_ok=True)
    blended_image.save(output_image_path)
    

    return output_image_path
   # return base_image_path

#not in use rn
def invert_base64_image(base64_str: str) -> str:
    # Decode the base64 string
    img_data = base64.b64decode(base64_str)

    # Convert the decoded data to a PIL Image object
    img = Image.open(io.BytesIO(img_data))

    # Invert the image colors
    inverted_img = ImageOps.invert(img)

    # Convert the inverted image back to a byte stream
    img_byte_arr = io.BytesIO()
    inverted_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Encode the inverted image as a base64 string
    inverted_base64_str = base64.b64encode(img_byte_arr).decode('utf-8')

    return inverted_base64_str

# hardens the mask
def harden_mask(encoded_image,taper):
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(decoded_image)).convert("RGBA")
    new_image = image
    # Make every pixel not black solid white
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
           # if r != 0 or g != 0 or b != 0:
           #     image.putpixel((x, y), (255, 255, 255, a))
            if r > 180 or g > 180 or b > 180:
                 new_image.putpixel((x, y), (255, 255, 255, a))

     #Taper for 3 pixels around
    if taper == True:
        new_image = new_image.filter(ImageFilter.BoxBlur(9))
        
        
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
           # if r != 0 or g != 0 or b != 0:
           #     image.putpixel((x, y), (255, 255, 255, a))
            if r > 180 or g > 180 or b > 180:
                 new_image.putpixel((x, y), (255, 255, 255, a))
     

    for x in range(width):
        for y in range(height):
            r, g, b, a = new_image.getpixel((x, y))
           # if r != 0 or g != 0 or b != 0:
           #     image.putpixel((x, y), (255, 255, 255, a))
            if r < 128 or g < 128 or b < 128:
                 new_image.putpixel((x, y), (r + 5, g + 5, b + 5,a))

    output_buffer = BytesIO()
    new_image.save(output_buffer, format="PNG")
    processed_image_base64 = base64.b64encode(output_buffer.getvalue()).decode()
   # combined = overlay_base64_images(encoded_image,processed_image_base64)
    return processed_image_base64


def resize_base64_image(base64_str, new_width: int, new_height: int) -> str:
    # Decode the base64 string
    #img_data = base64.b64decode(base64_str)
    #img_data2 = Image.fromarray( base64_to_texture(base64_str))
    # Convert the decoded data to a PIL Image object
    #img = Image.open(io.BytesIO(img_data))

    # Resize the image
    #resized_img = img_data2.resize((new_width, new_height), Image.ANTIALIAS)

    #decoded_data = base64.b64decode(base64_str)
    with open(base64_str, "rb") as f:
        bytes = f.read()

    # Create a BytesIO object
    buffered_data = io.BytesIO(bytes)

    # Open the image from the BytesIO object
    image = Image.open(buffered_data)
    resized_img = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Convert the resized image back to a byte stream
    img_byte_arr = io.BytesIO()
    resized_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Encode the resized image as a base64 string
    resized_base64_str = base64.b64encode(img_byte_arr).decode('utf-8')

    return resized_base64_str

def overlay_base64_images(encoded_image1, encoded_image2):
    decoded_image1 = base64.b64decode(encoded_image1)
    decoded_image2 = base64.b64decode(encoded_image2)

    image1 = Image.open(BytesIO(decoded_image1))
    image2 = Image.open(BytesIO(decoded_image2))

    # Overlay the images
    result = Image.blend(image1, image2,0.5)

    # Convert the overlaid image back to base64
    output_buffer = BytesIO()
    result.save(output_buffer, format="PNG")
    result_base64 = base64.b64encode(output_buffer.getvalue()).decode()

    return result_base64

#get all images inthe server
def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

# convert image to base64
# is this really th best way to do this?
def texture_to_base64(texture):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(texture).convert("RGBA")

    # Save the image to an in-memory buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # Get the byte data from the buffer and encode it as a base64 string
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return img_base64

# thanks to https://github.com/jinnsp ❤
def base64_to_texture(base64_string):
    if base64_string.lower().endswith('png'):
        image = Image.open(base64_string)
    else:
        decoded_data = base64.b64decode(base64_string)
        buffer = BytesIO(decoded_data)
        image = Image.open(buffer)
    texture = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return texture

def combine_masks(masks):
    """
    Combine three grayscale masks into one by overlaying them.
    
    Args:
        mask1 (np.ndarray): First grayscale mask with shape (height, width).
        mask2 (np.ndarray): Second grayscale mask with shape (height, width).
        mask3 (np.ndarray): Third grayscale mask with shape (height, width).
        
    Returns:
        np.ndarray: Combined grayscale mask with the same shape as the input masks.
    """
    combined_mask = np.maximum.reduce(masks)
    return combined_mask

def create_hole_mask(flow_map):
    h, w, _ = flow_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Compute the new coordinates of each pixel after the optical flow is applied
    new_x_coords = np.clip(x_coords + flow_map[..., 0], 0, w - 1).astype(int)
    new_y_coords = np.clip(y_coords + flow_map[..., 1], 0, h - 1).astype(int)

    # Create a 2D array to keep track of whether a pixel is occupied or not
    occupied = np.zeros((h, w), dtype=bool)

    # Mark the pixels that are occupied after the optical flow is applied
    occupied[new_y_coords, new_x_coords] = True

    # Create the hole mask by marking unoccupied pixels as holes (value of 1)
    hole_mask = np.logical_not(occupied).astype(np.uint8)

    

    expanded = filter_mask(hole_mask) * 255
    #expanded = hole_mask * 255
    #blurred_hole_mask = box_(expanded, sigma=3)
    toblur = Image.fromarray(expanded).convert('L')
    blurred_hole_mask = np.array(toblur.filter(ImageFilter.GaussianBlur(3)))

    #blurred_numpy = np.array( Image.fromarray(expanded).filter(ImageFilter.GaussianBlur(3)))
    #blurred_hole_mask[blurred_hole_mask > 150] = 255
    filtered_smol = filter_mask(hole_mask,4,0.4,0.3) * 255
    return blurred_hole_mask + filtered_smol

# there are pixels all over the place that are not holes, so this only gets the holes with a high concentration
def filter_mask(mask, kernel_size=4, threshold_ratio=0.3,grayscale_intensity=1.0):
    # Create a custom kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Convolve the mask with the kernel
    conv_result = cv2.filter2D(mask, -1, kernel)

    # Calculate the threshold based on the ratio
    threshold = int(kernel.size * threshold_ratio)

    # Filter the mask using the calculated threshold
    filtered_mask = np.where(conv_result >= threshold, mask, 0)

    # thanks to https://github.com/jinnsp ❤
    grayscale_mask = np.where(conv_result >= threshold, int(255 * grayscale_intensity), 0).astype(np.uint8)

    # Combine the filtered mask and grayscale mask
    combined_mask = np.maximum(filtered_mask, grayscale_mask)


    return combined_mask


def resize_image(image, max_dimension_x, max_dimension_y):
    h, w = image.shape[:2]
    aspect_ratio = float(w) / float(h)
    if h > w:
        new_height = max_dimension_y
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension_x
        new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

#delete this later
def replaced_mask_from_other_direction_debug(index,image, mask, flowmap, original, output_folder1='./debug',forwards=True):
    # Ensure inputs are numpy arrays
    image = np.array(image)
    mask = np.array(mask)
    
    flowmap_detached = flowmap.detach().cpu().numpy()
    if original is not None:
        original = np.array(original)
        original = resize_image(original, image.shape[1], image.shape[0])



    # Get the height and width of the image
    height, width, _ = image.shape

    # Get the coordinates of the masked pixels
    masked_coords = np.argwhere(mask)

    # Initialize a new_image to store the result
    new_image = image.copy()

    # Initialize debug_image to store debug information


    for y, x in masked_coords:
        border_limit = 10
        # Skip if pixel is within 20 pixels of the border
        if border_limit <= y < height - border_limit and border_limit <= x < width - border_limit:

            # Get the opposite flow vector
            flow_vector = np.array([flowmap_detached[0, y, x], flowmap_detached[1, y, x]])

            # Calculate the new coordinates after applying the flow
            new_y = int(round(y - flow_vector[1] * 2))
            new_x = int(round(x - flow_vector[0] * 2))
            
            # Check if the new coordinates are inside the image bounds
            if 0 <= new_y < height and 0 <= new_x < width:
                # Replace the pixel in the new_image with the pixel from the opposite flow direction
                intensity = mask[y, x] / 255
                weight = gaussian_weight(intensity, sigma=0.5)

                # Replace the pixel in the new_image with the pixel from the opposite flow direction, weighted by the Gaussian weight
                #new_image[y, x] = weight * image[new_y, new_x] + (1 - weight) * image[y, x]
                #if original is not None and not is_similar_color(image[y, x], original[y, x],50):
                    # weird edge case where if the background is moving and the foreground is not, the foreground will be replaced by the background
                ###    new_image[y, x] = weight * image[y, x] + (1 - weight) * original[y, x]
                #else:
                new_image[y, x] = (1 - weight) * image[new_y, new_x] + weight * image[y, x]
                
                
                # Replace the pixel in the new_image with the pixel from the opposite flow direction, weighted by the intensity
                #new_image[y, x] = intensity * image[new_y, new_x] + (1 - intensity) * image[y, x]
                #new_image[y, x] = image[new_y, new_x]
                #new_image[y, x] = image[new_y, new_x] + (1 - intensity) * image[y, x]
                # Update debug_image
                #debug_image[y, x] = [0,0,255 * (1 - weight)]  # Red color for the mask
                #debug_image[new_y, new_x] = [0, 255 * (1 - weight),0]  # Green color for the moved pixel content
                #debug_image[y, x] = [0,0,255 * (1 - weight)]  # Blue color for the moved pixel destination


    #output_folder = os.path.join(output_folder1, f'newmaskimg{index}.png')
    # Save the debug image to the specified folder


    #if original is not None:
    #debug_image_pil = Image.fromarray(debug_image)
    #debug_image_pil.save(output_folder)

    return texture_to_base64(new_image)


def is_similar_color(pixel1, pixel2, threshold):
    """
    Returns True if pixel1 and pixel2 are similar in color within the given threshold, False otherwise.
    """
    r1, g1, b1,a1 = pixel1
    r2, g2, b2,a2 = pixel2
    color_difference = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    return color_difference <= threshold

def gaussian_weight(d, sigma=1.0):
    return np.exp(-(d ** 2) / (2 * sigma ** 2))

def avg_edge_pixels(img):
    height, width = img.shape[:2]
    edge_pixels = []
    
    # top and bottom edges
    edge_pixels.extend(img[0,:])
    edge_pixels.extend(img[height-1,:])
    
    # left and right edges
    edge_pixels.extend(img[:,0])
    edge_pixels.extend(img[:,width-1])
    
    # calculate average of edge pixels
    avg_edge_pixel = np.mean(edge_pixels)
    
    return avg_edge_pixel

def check_edges(image):
    h, w, c = image.shape
    threshold = 0.4 
    
    def is_different(pixel1, pixel2):
        return np.any(np.abs(pixel1 - pixel2) > threshold * 255)

    for i in range(h):
        for j in range(w):
            if i < 2 or i > h - 3 or j < 2 or j > w - 3:
                central_i = i + 5 if i < h // 2 else i - 5
                central_j = j + 5 if j < w // 2 else j - 5
                
                # Ensure the central pixel is within the image boundaries
                central_i = max(0, min(central_i, h - 1))
                central_j = max(0, min(central_j, w - 1))

                if is_different(image[i, j], image[central_i, central_j]):
                    image[i, j] = image[central_i, central_j]


def resize_to_nearest_multiple_of_8(width, height):
    def nearest_multiple(n, factor):
        return round(n / factor) * factor

    new_width = nearest_multiple(width, 8)
    new_height = nearest_multiple(height, 8)
    return new_width, new_height



def resize_to_nearest_multiple(width, height, a):
    def nearest_common_multiple(target, a, b):
        multiple = 1
        nearest_multiple = 0
        min_diff = float('inf')

        while True:
            current_multiple = a * multiple
            if current_multiple % b == 0:
                diff = abs(target - current_multiple)
                if diff < min_diff:
                    min_diff = diff
                    nearest_multiple = current_multiple
                else:
                    break
            multiple += 1

        return nearest_multiple

    new_width = nearest_common_multiple(width, a, 8)
    new_height = nearest_common_multiple(height, a, 8)
    return int(new_width), int(new_height)


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


def pil_images_to_video(pil_images, output_file, fps=24):
    """
    Saves an array of PIL images to a video file using MoviePy.

    Args:
        pil_images (list): A list of PIL images.
        output_file (str): The output file path for the video.
        fps (int, optional): The desired frames per second. Defaults to 24.

    Returns:
        the filepath of the video file
    """
    # Convert PIL images to NumPy arrays
    np_images = [np.array(img) for img in pil_images]

    # Create an ImageSequenceClip instance with the array of NumPy images and the specified fps
    clip = ImageSequenceClip(np_images, fps=fps)

    # Write the video file to the specified output location
    clip.write_videofile(output_file,fps,codec='libx264')

    return output_file

def copy_video(source_path, destination_path):
    """
    Copy a video file from source_path to destination_path.

    :param source_path: str, path to the source video file
    :param destination_path: str, path to the destination video file
    :return: None
    """
    try:
        shutil.copy(source_path, destination_path)
        print(f"Video copied successfully from {source_path} to {destination_path}")
    except IOError as e:
        print(f"Unable to copy video. Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def crossfade_frames(frame1, frame2, alpha):
    """Crossfade between two video frames with a given alpha value."""
    image1 = Image.fromarray(frame1)
    image2 = Image.fromarray(frame2)
    blended_image = crossfade_images(image1, image2, alpha)
    blended_image = blended_image.convert('RGB')
    #THIS FUNCTION CONSUMED 3 HOURS OF DEBUGING AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    return np.array(blended_image)
       



def crossfade_videos(video_paths,fps, overlap_indexes, num_overlap_frames, output_path):

    """
        a python function and i need it to input an array of video paths,
        an array of the indexes where the videos overlap with the next video,
        the number of overlapping frames and the next file and i need it to combine these
        clips while crossfading between the clips where there is overlapping happening and 
        output it to the output file location
    """

    #not video paths any more frame arrays
    original_frames_arrays = video_paths
    #for i in range(len(video_paths)):
    #    data = convert_video_to_bytes(video_paths[i])
    ##    frames_list = extract_frames_movpie(data, fps)
    #    original_frames_arrays.append(frames_list)
    #    print (f"video {i} has {len(frames_list)} frames")
    new_frames_arrays = copy.deepcopy(original_frames_arrays)
    
    for index, frames_array in enumerate(original_frames_arrays):
        if index < len(original_frames_arrays) - 1 and index in overlap_indexes:
            next_array = original_frames_arrays[index+1]
            print (f"crossfading between video {index} and video {index+1}")
            first_of_next = next_array[:num_overlap_frames]
            last_of_current = frames_array[-num_overlap_frames:]
            #last_of_current = last_of_current[::-1]
            if len(first_of_next) != len(last_of_current):
                print ("crossfade frames are not the same length")
                while len(first_of_next) != len(last_of_current):
                    if len(first_of_next) > len(last_of_current):
                        first_of_next.pop()  # remove the last element from array1
                    else:
                        last_of_current.pop()  # remove the last element from array2
                
            crossfaded = []
            for i in range(num_overlap_frames):
                alpha = 1 - (i / num_overlap_frames) # set alpha value
                if i > len(last_of_current) - 1 or i > len(first_of_next) - 1:
                    
                    print ("ran out of crossfade space at index ",str(i))
                    break;
                
                new_frame = crossfade_frames(last_of_current[i], first_of_next[i], alpha)
                #print (new_frame.shape)
                crossfaded.append(new_frame)
            print (f"crossfaded {len(crossfaded)} frames with num overlap = {num_overlap_frames}, the last of current array is of length {len(last_of_current)} and the first of next is of length {len(first_of_next)}")
            #saving first of next and last of current
            new_frames_arrays[index][-num_overlap_frames:] = crossfaded

        if index > 0 and index - 1 in overlap_indexes:
            new_frames_arrays[index] = new_frames_arrays[index][num_overlap_frames:]

    for arr in new_frames_arrays:
        print(len(arr))
    #combined_arrays = np.concatenate(new_frames_arrays)
    output_array = []
    for arr in new_frames_arrays:
        for frame in arr:
            #frame =  cv2.resize(frame, (new_frames_arrays, new_height), interpolation=cv2.INTER_LINEAR)
            #print (frame.shape)
            output_array.append(Image.fromarray(frame).convert("RGB"))
    return pil_images_to_video(output_array, output_path, fps)




def crossfade_images(image1, image2, alpha):
    """Crossfade between two images with a given alpha value."""
    image1 = image1.convert("RGBA")
    image2 = image2.convert("RGBA")
    return Image.blend(image1, image2, alpha)


def extract_frames_movpie(input_video, target_fps, max_frames=None, perform_interpolation=True):
    print("Interpolating extra frames")

    def get_video_info(video_path):
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout)

    def interpolate_frames(frame1, frame2, ratio):
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return cv2.addWeighted(frame1, 1 - ratio, frame2, ratio, 0) + ratio * cv2.remap(frame1, flow * (1 - ratio), None, cv2.INTER_LINEAR)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(input_video)
    video_path = f.name

    video_info = get_video_info(video_path)
    video_stream = next((stream for stream in video_info['streams'] if stream['codec_type'] == 'video'), None)
    original_fps = float(video_stream['avg_frame_rate'].split('/')[0])

    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration

    frames = []
    frame_ratio = original_fps / target_fps
    frame_time = 1 / target_fps
    current_time = 0


    while current_time < video_duration and (max_frames is None or len(frames) < max_frames):
        frame1_time = current_time * frame_ratio
        frame2_time = min((current_time + frame_time) * frame_ratio, video_duration)
        frame1 = video_clip.get_frame(frame1_time)
        frame2 = video_clip.get_frame(frame2_time)

        if not perform_interpolation or target_fps <= original_fps:
            frame = frame1
        else:
            ratio = (frame1_time * original_fps) % 1
            frame = interpolate_frames(frame1, frame2, ratio)

        frames.append(frame)
        current_time += frame_time

    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]

    print(f"Extracted {len(frames)} frames at {target_fps} fps over a clip with a length of {len(frames) / target_fps} seconds with the old duration of {video_duration} seconds")
    return frames

def convert_video_to_bytes(input_file):
    # Read the uploaded video file
    print(f"reading video file... {input_file}")
    with open(input_file, "rb") as f:
        video_bytes = f.read()

    # Return the processed video bytes (or any other output you want)
    return video_bytes

def split_video_into_numpy_arrays(video_path, target_fps=None, perform_interpolation=False):

    video_manager = scenedetect.VideoManager([video_path])
    scene_manager = scenedetect.SceneManager()
    scene_manager.add_detector(scenedetect.ContentDetector())

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    
    if target_fps is not None:
        original_fps = video_manager.get(cv2.CAP_PROP_FPS)

    if len(scene_list) == 0:
        start_time = 0
        end_time = video_manager.get(cv2.CAP_PROP_FRAME_COUNT) / video_manager.get(cv2.CAP_PROP_FPS)
        scene_list.append((start_time, end_time))

    print (f"Detected {len(scene_list)} scenes")
    numpy_arrays = save_scenes_as_numpy_arrays(scene_list, video_path, target_fps, original_fps if target_fps else None, perform_interpolation)

    print(f"Total scenes: {len(numpy_arrays)}")
    return numpy_arrays

def save_scenes_as_numpy_arrays(scene_list, video_path, target_fps=None, original_fps=None, perform_interpolation=True):
    def interpolate_frames(frame1, frame2, ratio):
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return cv2.addWeighted(frame1, 1 - ratio, frame2, ratio, 0) + ratio * cv2.remap(frame1, flow * (1 - ratio), None, cv2.INTER_LINEAR)

    numpy_arrays = []
    video_capture = cv2.VideoCapture(video_path)
    
    if target_fps and original_fps:
        frame_ratio = original_fps / target_fps
        frame_time = 1 / target_fps

    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = int(start_time.get_frames())
        end_frame = int(end_time.get_frames())
        scene_frames = []
        current_time = start_frame / original_fps if target_fps else start_frame

        print(f"Processing scene {i + 1}: start_frame={start_frame}, end_frame={end_frame} original fps={original_fps} target fps={target_fps}")

        while current_time < end_frame / original_fps if target_fps else end_frame:
            if target_fps and original_fps and perform_interpolation:
                frame1_time = current_time * frame_ratio
                frame2_time = min((current_time + frame_time) * frame_ratio, end_frame / original_fps)
            else:
                frame1_time = current_time
                frame2_time = current_time

            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame1_time * original_fps if target_fps else frame1_time)
            ret, frame1 = video_capture.read()

            if ret:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            if target_fps and original_fps and perform_interpolation:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame2_time * original_fps)
                ret, frame2 = video_capture.read()

                if ret:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                ratio = (frame1_time * original_fps) % 1
                frame = interpolate_frames(frame1, frame2, ratio)
            else:
                frame = frame1

            scene_frames.append(frame)
            
            current_time += frame_time if target_fps else 1
        print(f"Scene {i + 1} has {len(scene_frames)} frames with length of {len(scene_frames)} frames with the old duration of {end_frame - start_frame} frames")
        numpy_arrays.append(np.array(scene_frames))

    video_capture.release()
    return numpy_arrays