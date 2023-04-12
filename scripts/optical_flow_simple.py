import os
import glob
import cv2
import numpy as np
from PIL import Image
import scripts.berry_utility as utilityb

def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image '{file_path}' not found.")
    return image
    
def save_image(image, file_path):
    cv2.imwrite(file_path, image)

def resize_image(image, max_dimension):
    h, w = image.shape[:2]
    aspect_ratio = float(w) / float(h)
    if h > w:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image
    
def flow_to_color(flow, max_flow=None):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 1] = 1.0

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    if max_flow is not None:
        hsv[..., 2] = np.clip(mag / max_flow, 0, 1)
    else:
        hsv[..., 2] = np.clip(mag / (np.max(mag) + 1e-5), 0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (rgb * 255).astype(np.uint8)

def save_optical_flow(flow, file_path, max_flow=None):
    flow_color = flow_to_color(flow, max_flow)
    save_image(flow_color, file_path)


def compute_optical_flow(image1, image2):
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow

def warp_image(image, flow):
    h, w = image.shape[:2]
    flow_map = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32) - flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)  # Ensure the flow_map is in the correct format

    # Clip the flow_map to the image bounds
    flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, w - 1)
    flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, h - 1)

    warped_image = cv2.remap(image, flow_map, None, cv2.INTER_LANCZOS4    )
    return warped_image

def process_image_basic (image1_path, image2_path, provided_image_path,max_dimension, index,output_folder):

    #image1 = read_image(image1_path)
   # image2 = read_image(image2_path)
    
#    image1 = resize_image(image1, max_dimension)
 #   image2 = resize_image(image2, max_dimension)
    image1 =  resize_image(utilityb.base64_to_texture(image1_path),max_dimension)
    image2 =  resize_image(utilityb.base64_to_texture(image2_path),max_dimension)
    
#    provided_image = read_image(provided_image_path)
    provided_image = utilityb.base64_to_texture(provided_image_path)
    provided_image = resize_image(provided_image, max_dimension)
    
    flow = compute_optical_flow(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
    warped_image = warp_image(provided_image, flow)

    warped_image_path = os.path.join(output_folder, f'warped_provided_image_{index + 1}.png')
    save_image(warped_image, warped_image_path)
    print(f"Warped image saved as '{warped_image_path}'")
    combine_images(image1,image2,provided_image,warped_image,f"{index}.png")
    return warped_image_path,flow
  #  return provided_image_path,flow
    #AAAAAAAAAAAAAAAAAAA

def process_image(image1, image2, provided_image, output_folder, flow_output_folder, max_dimension, index):
    image1 = resize_image(image1, max_dimension)
    image2 = resize_image(image2, max_dimension)

    flow = compute_optical_flow(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
    warped_image = warp_image(provided_image, flow)

    warped_image_path = os.path.join(output_folder, f'warped_provided_image_{index + 1}.png')
    save_image(warped_image, warped_image_path)
    print(f"Warped image saved as '{warped_image_path}'")

    flow_image_path = os.path.join(flow_output_folder, f'optical_flow_{index + 1}.png')
    save_optical_flow(flow, flow_image_path)
    print(f"Optical flow map saved as '{flow_image_path}'")

    return warped_image

def process_images(input_folder, output_folder, flow_output_folder, provided_image_path, max_dimension):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(flow_output_folder):
        os.makedirs(flow_output_folder)

    image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')))
    num_images = len(image_files)

    if num_images < 1:
        raise ValueError("At least one image is required to compute optical flow.")

    provided_image = read_image(provided_image_path)
    provided_image = resize_image(provided_image, max_dimension)

    for i in range(num_images - 1):
        image1 = read_image(image_files[i])
        image2 = read_image(image_files[i + 1])

        provided_image = process_image(image1, image2, provided_image, output_folder, flow_output_folder, max_dimension, i)

def main():
    input_folder = "Input_Images"
    output_folder = "output_images"
    flow_output_folder = "flow_output_images"
    provided_image_path = "init.png"
    max_dimension = 320 # Change this value to your preferred maximum dimension
    #process_images(input_folder, output_folder, flow_output_folder, provided_image_path, max_dimension)


if __name__ == "__main__":
    main()
    
def combine_images(img1, img2, img3, img4, output_file, output_folder="debug"):
    img1_pil = Image.fromarray(np.uint8(img1))
    img2_pil = Image.fromarray(np.uint8(img2))
    img3_pil = Image.fromarray(np.uint8(img3))
    img4_pil = Image.fromarray(np.uint8(img4))

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Combine images horizontally
    combined_horizontal1 = Image.new('RGB', (img1_pil.width + img2_pil.width, img1_pil.height))
    combined_horizontal1.paste(img1_pil, (0, 0))
    combined_horizontal1.paste(img2_pil, (img1_pil.width, 0))

    combined_horizontal2 = Image.new('RGB', (img3_pil.width + img4_pil.width, img3_pil.height))
    combined_horizontal2.paste(img3_pil, (0, 0))
    combined_horizontal2.paste(img4_pil, (img3_pil.width, 0))

    # Combine images vertically
    combined_image = Image.new('RGB', (combined_horizontal1.width, combined_horizontal1.height + combined_horizontal2.height))
    combined_image.paste(combined_horizontal1, (0, 0))
    combined_image.paste(combined_horizontal2, (0, combined_horizontal1.height))

    # Save combined image to output folder
    output_path = os.path.join(output_folder, output_file)
    combined_image.save(output_path)

