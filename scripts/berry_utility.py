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

def base64_to_texture(base64_string):
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

    grayscale_mask = np.where(conv_result >= threshold, int(255 * grayscale_intensity), 0)

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