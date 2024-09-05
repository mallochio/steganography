import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

def lsb_encode(cover_image, secret_image):
    """
    Encode a secret image into a cover image using LSB steganography.
    
    Args:
        cover_image: PIL Image or numpy array of cover image
        secret_image: PIL Image or numpy array of secret image (should be binary)
    
    Returns:
        PIL Image with encoded data
    """
    if isinstance(cover_image, Image.Image):
        cover_image = np.array(cover_image)
    if isinstance(secret_image, Image.Image):
        secret_image = np.array(secret_image)
        
    # Ensure secret image is binary (0 or 1)
    if secret_image.dtype != np.uint8 or secret_image.max() > 1:
        secret_image = (secret_image > 127).astype(np.uint8)
    
    # Resize secret image to match cover image dimensions
    if secret_image.shape != cover_image.shape:
        secret_image = np.array(Image.fromarray(secret_image).resize(
            (cover_image.shape[1], cover_image.shape[0])))
    
    # Encode the secret image in the least significant bit
    encoded_image = cover_image & ~1 | secret_image
    
    return Image.fromarray(encoded_image)

def lsb_decode(stego_image):
    """
    Decode a secret image from a stego image using LSB steganography.
    
    Args:
        stego_image: PIL Image or numpy array of stego image
    
    Returns:
        PIL Image with decoded data
    """
    if isinstance(stego_image, Image.Image):
        stego_image = np.array(stego_image)
        
    # Extract the least significant bit
    decoded_data = stego_image & 1
    
    # Convert to visible image (multiply by 255 to make it visible)
    decoded_image = decoded_data * 255
    
    return Image.fromarray(decoded_image.astype(np.uint8))

def dct_encode(cover_image, secret_image, scale_factor=0.1, block_size=8):
    """
    Encode a secret image into a cover image using DCT-based steganography.
    
    Args:
        cover_image: Grayscale numpy array of cover image
        secret_image: Grayscale numpy array of secret image
        scale_factor: Factor to scale secret image (controls visibility)
        block_size: Size of DCT blocks
    
    Returns:
        Numpy array with encoded image
    """
    if isinstance(cover_image, Image.Image):
        cover_image = np.array(cover_image)
    if isinstance(secret_image, Image.Image):
        secret_image = np.array(secret_image)
        
    # Normalize secret image
    secret_image = secret_image / 255.0 * scale_factor
    
    # Perform DCT on cover image in blocks
    cover_image_dct = np.zeros_like(cover_image, dtype=float)
    for i in range(0, cover_image.shape[0], block_size):
        for j in range(0, cover_image.shape[1], block_size):
            block = cover_image[i:min(i+block_size, cover_image.shape[0]), 
                                j:min(j+block_size, cover_image.shape[1])]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                cover_image_dct[i:(i+block_size), j:(j+block_size)] = dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    # Insert secret image into mid-frequency DCT coefficients
    h, w = secret_image.shape[:2]
    cover_image_dct[block_size//2:block_size//2+h, block_size//2:block_size//2+w] += secret_image
    
    # Perform inverse DCT to get stego image
    stego_image = np.zeros_like(cover_image_dct, dtype=float)
    for i in range(0, cover_image_dct.shape[0], block_size):
        for j in range(0, cover_image_dct.shape[1], block_size):
            block = cover_image_dct[i:min(i+block_size, cover_image_dct.shape[0]), 
                                    j:min(j+block_size, cover_image_dct.shape[1])]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                stego_image[i:(i+block_size), j:(j+block_size)] = idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    # Clip and convert back to uint8
    stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
    
    return stego_image

def dct_decode(stego_image, cover_image, scale_factor=0.1, block_size=8, secret_shape=None):
    """
    Decode a secret image from a stego image using DCT-based steganography.
    
    Args:
        stego_image: Numpy array of stego image
        cover_image: Original cover image used for encoding
        scale_factor: Factor used during encoding
        block_size: Size of DCT blocks
        secret_shape: Shape of the original secret image (optional)
    
    Returns:
        Numpy array with decoded secret image
    """
    if isinstance(stego_image, Image.Image):
        stego_image = np.array(stego_image)
    if isinstance(cover_image, Image.Image):
        cover_image = np.array(cover_image)
        
    # Perform DCT on both images
    stego_dct = np.zeros_like(stego_image, dtype=float)
    cover_dct = np.zeros_like(cover_image, dtype=float)
    
    for i in range(0, stego_image.shape[0], block_size):
        for j in range(0, stego_image.shape[1], block_size):
            if i+block_size <= stego_image.shape[0] and j+block_size <= stego_image.shape[1]:
                stego_dct[i:(i+block_size), j:(j+block_size)] = dct(dct(stego_image[i:(i+block_size), j:(j+block_size)].T, norm='ortho').T, norm='ortho')
                cover_dct[i:(i+block_size), j:(j+block_size)] = dct(dct(cover_image[i:(i+block_size), j:(j+block_size)].T, norm='ortho').T, norm='ortho')
    
    # Extract the difference which contains the secret image
    if secret_shape is None:
        # Default to half of the cover image size
        h, w = stego_image.shape[0] // 2, stego_image.shape[1] // 2
    else:
        h, w = secret_shape
        
    # Extract the secret from mid-frequency coefficients
    decoded_secret = stego_dct[block_size//2:block_size//2+h, block_size//2:block_size//2+w] - \
                    cover_dct[block_size//2:block_size//2+h, block_size//2:block_size//2+w]
    
    # Scale back and convert to image format
    decoded_secret = decoded_secret / scale_factor
    decoded_secret = np.clip(decoded_secret * 255, 0, 255).astype(np.uint8)
    
    return decoded_secret
