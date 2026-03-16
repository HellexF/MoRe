# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import torch.nn.functional as F
import numpy as np


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def load_and_preprocess_masks(image_path_list, mode="crop"):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # 转换为灰度图（白色填充透明区域）
        img = img.convert("L")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def preprocess_image_and_depth(
    image,
    depth_map,
    intri=None, 
    mode="crop",
    target_size=518,
    divisible=14,
    pad_value_depth=0.0, 
    pad_value_image=1.0 
):
    if mode not in ["crop", "pad"]:
        raise ValueError("mode must be 'crop' or 'pad'")
    if isinstance(image, torch.Tensor):
        img_t = image.clone()
        if img_t.ndim == 3 and img_t.shape[0] in (1,3):  # (C,H,W)
            pass
        elif img_t.ndim == 3 and img_t.shape[-1] in (1,3):  # (H,W,C)
            img_t = img_t.permute(2,0,1)
        elif img_t.ndim == 2:  # (H,W)
            img_t = img_t.unsqueeze(0)  # (1,H,W)
        else:
            raise ValueError(f"Unsupported image shape: {tuple(img_t.shape)}")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[-1] in (1,3):       # (H,W,C)
            img_t = torch.from_numpy(arr).permute(2,0,1)
        elif arr.ndim == 2:                                 # (H,W)
            img_t = torch.from_numpy(arr).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
    img_t = img_t.float().unsqueeze(0)  # (1,C,H,W)

    # depth
    if isinstance(depth_map, torch.Tensor):
        d_t = depth_map.clone()
        if d_t.ndim != 2:
            raise ValueError(f"Depth must be 2D (H,W), got {tuple(d_t.shape)}")
    else:
        d_t = torch.from_numpy(np.asarray(depth_map))
        if d_t.ndim != 2:
            raise ValueError(f"Depth must be 2D (H,W), got {tuple(d_t.shape)}")
    d_t = d_t.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    H, W = img_t.shape[-2], img_t.shape[-1]
    if mode == "pad":
        if W >= H:
            new_w = target_size
            new_h = max(divisible, int(round((H * (new_w / W)) / divisible) * divisible))
        else:
            new_h = target_size
            new_w = max(divisible, int(round((W * (new_h / H)) / divisible) * divisible))
    else:  # crop
        new_w = target_size
        new_h = max(divisible, int(round((H * (new_w / W)) / divisible) * divisible))

    img_resized = F.interpolate(img_t, size=(new_h, new_w), mode="bicubic", align_corners=False)
    depth_resized = F.interpolate(d_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

    if mode == "crop" and new_h > target_size:
        start_y = (new_h - target_size) // 2
        img_final = img_resized[:, :, start_y:start_y + target_size, :]           # (1,C,518,518)
        depth_final = depth_resized[:, :, start_y:start_y + target_size, :]       # (1,1,518,518)
        pad_l = pad_r = pad_t = pad_b = 0
    else:
        h_padding = target_size - new_h
        w_padding = target_size - new_w
        pad_t = max(0, h_padding // 2)
        pad_b = max(0, h_padding - pad_t)
        pad_l = max(0, w_padding // 2)
        pad_r = max(0, w_padding - pad_l)
        img_final = F.pad(img_resized, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=pad_value_image)
        depth_final = F.pad(depth_resized, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=pad_value_depth)

    intri_out = None
    if intri is not None:
        sx = new_w / W
        sy = new_h / H

        def update_K(K):
            K2 = K.copy()
            K2[0][0] *= sx  # fx
            K2[1][1] *= sy  # fy
            K2[0][2] *= sx  # cx
            K2[1][2] *= sy  # cy
            if mode == "crop" and new_h > target_size:
                K2[1][2] -= (new_h - target_size) // 2
            else:
                K2[0][2] += pad_l
                K2[1][2] += pad_t
            return K2
        intri_out = update_K(intri)
    
    img_out = img_final.squeeze(0)          # (C,518,518)
    depth_out = depth_final.squeeze(0).squeeze(0)  # (518,518)
    return img_out, depth_out, intri_out