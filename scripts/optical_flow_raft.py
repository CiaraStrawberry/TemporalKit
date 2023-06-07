import cv2
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

#no clue if this works
def flow_to_rgb(flow):
    """
    Convert optical flow to RGB image
    
    :param flow: optical flow map
    :return: RGB image
    
    """
    # forcing conversion to float32 precision
    flow = flow.numpy()
    hsv = np.zeros(flow.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #cv2.imshow("colored flow", bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return hsv

def write_flo(flow, filename):
    """
    Write optical flow in Middlebury .flo format
    
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    
    from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    # forcing conversion to float32 precision
    flow = flow.cpu().data.numpy()
    flow = flow.astype(np.float32)
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


#def infer_old (frameA,frameB)

def infer(frameA, frameB):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    # Check if both frames have the same size
    if frameA.size != frameB.size:
        raise ValueError("Both input frames must have the same size")

    transform = T.ToTensor()

    img1_batch = transform(frameA)
    img2_batch = transform(frameB)
    img1_batch = torch.stack([img1_batch])
    img2_batch = torch.stack([img2_batch])
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    def preprocess(img1_batch, img2_batch):
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    return img1_batch, img2_batch
 


def apply_flow_based_on_images (image1_path, image2_path, provided_image_path,max_dimension, index,output_folder,midas):
    w,h = get_target_size(utilityb.base64_to_texture(image1_path), max_dimension)
    w =  int(w / 8) * 8
    h =  int(h / 8) * 8
    image1 =  resize_image(utilityb.base64_to_texture(image1_path),h,w)
    h, w = image1.shape[:2]
    image2 =  cv2.resize(utilityb.base64_to_texture(image2_path), (w,h), interpolation=cv2.INTER_LINEAR)


    provided_image = utilityb.base64_to_texture(provided_image_path)
    provided_image = cv2.resize(provided_image, (w,h), interpolation=cv2.INTER_LINEAR)
    


    img1_batch,img2_batch = infer(image1,image2)
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flow = list_of_flows[-1][0]
    flow_img = flow_to_image(predicted_flow).to("cpu")

    warped_image,unused_mask,white_pixels = apply_flow_to_image_with_unused_mask(provided_image,predicted_flow)


    warped_image_path = os.path.join(output_folder, f'warped_provided_image_{index + 1}.png')
    save_image(warped_image, warped_image_path)
    return warped_image_path,predicted_flow,unused_mask,white_pixels,flow_img

def apply_flow_to_image(image, flow):
    """
    Apply optical flow transforms to an input image
    
    :param image: input image
    :param flow: optical flow map
    :return: warped image
    
    """
    
    # forcing conversion to float32 precision
    #flow = flow.numpy()
    flow = flow.astype(np.float32)

    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Create a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply the optical flow to the coordinates
    x_warped = (x + flow[..., 0]).astype(np.float32)
    y_warped = (y + flow[..., 1]).astype(np.float32)

    # Remap the input image using the warped coordinates
    warped_image = cv2.remap(image, x_warped, y_warped, cv2.INTER_LINEAR)

    return warped_image

def warp_image(image, flow):
    h, w = image.shape[:2]

    flow_map = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32) - flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)  # Ensure the flow_map is in the correct format

    # Clip the flow_map to the image bounds
    flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, w - 1)
    flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, h - 1)

    warped_image = cv2.remap(image, flow_map, None, cv2.INTER_LANCZOS4)
    return warped_image

def save_image(image, file_path):
    cv2.imwrite(file_path, image)

def resize_image(image, new_height,new_width):
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def get_target_size (image,max_dimension):
    h, w = image.shape[:2]
    aspect_ratio = float(w) / float(h)
    if h > w:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    return new_width,new_height

        
def apply_flow_to_image_try3(image,flow):
    """
    Apply an optical flow tensor to a NumPy image by moving the pixels based on the flow.
    
    Args:
        image (np.ndarray): Input image with shape (height, width, channels).
        flow (np.ndarray): Optical flow tensor with shape (height, width, 2).
        
    Returns:
        np.ndarray: Warped image with the same shape as the input image.
    """
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

    # Add the flow to the original coordinates
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = flow.transpose(1, 2, 0)
    new_coords = np.subtract(coords, flow)


    # Map the new coordinates to the pixel values in the original image
    warped_image = cv2.remap(image, new_coords, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image


def apply_flow_to_image_with_unused_mask(image, flow,depth_map):
    """
    Apply an optical flow tensor to a NumPy image by moving the pixels based on the flow and create a mask where the remap meant there was nothing there.
    
    Args:
        image (np.ndarray): Input image with shape (height, width, channels).
        flow (np.ndarray): Optical flow tensor with shape (height, width, 2).
        
    Returns:
        tuple: Warped image with the same shape as the input image, and a mask where the remap meant there was nothing there.
    """
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

    # Add the flow to the original coordinates
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = flow.transpose(1, 2, 0)
    new_coords = np.subtract(coords, flow)
    warped_image = cv2.remap(image, new_coords, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Create a mask where the remap meant there was nothing there
    mask = utilityb.create_hole_mask(flow)
    white_pixels = np.sum(mask > 0)

    return warped_image, mask,white_pixels

def warp_image2(image, flow):
    h, w = image.shape[:2]
    flow_map = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32) - flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)  # Ensure the flow_map is in the correct format

    # Clip the flow_map to the image bounds
    flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, w - 1)
    flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, h - 1)

    warped_image = cv2.remap(image, flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0, 0, 0)    )
    return warped_image

def remap_with_depth(image, flow, depth_map):
    """
    Remap an image according to the optical flow and depth map.

    Args:
        image (np.ndarray): Input image with shape (height, width, channels).
        flow (np.ndarray): Optical flow tensor with shape (height, width, 2).
        depth_map (np.ndarray): Depth map with shape (height, width).

    Returns:
        np.ndarray: Remapped image with the same shape as the input image.
    """
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
    
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = flow.transpose(1, 2, 0)

    new_coords = np.subtract(coords, flow)
    new_coords = new_coords.astype(np.int)

    new_coords[..., 0] = np.clip(new_coords[..., 0], 0, width - 1)
    new_coords[..., 1] = np.clip(new_coords[..., 1], 0, height - 1)

    remapped_image = np.zeros_like(image)
    depth_buffer = np.zeros_like(depth_map)

    for y in range(height):
        for x in range(width):
            new_x, new_y = new_coords[y, x]
            if depth_map[y, x] > depth_buffer[new_y, new_x]:
                remapped_image[new_y, new_x] = image[y, x]
                depth_buffer[new_y, new_x] = depth_map[y, x]
                
    return remapped_image