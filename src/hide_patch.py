#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   hide_patch.py
@Time    :   2023/05/13 18:04:35
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Steganography tool to hide and recover private patches in images
'''

import cv2
import torch
import zlib
import math
import numpy as np
import argparse
import sys
from PIL import Image
import reedsolo
from cryptography.fernet import Fernet
from torchvision.transforms import ToTensor
from torchvision.models.segmentation import deeplabv3_resnet50


def generate_encryption_key():
    return Fernet.generate_key()


def encrypt_data(data, key):
    f = Fernet(key)
    return f.encrypt(data)


def decrypt_data(data, key):
    f = Fernet(key)
    return f.decrypt(data)


def load_deeplab_model():
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model


def predict_person_mask(image, model):
    input_tensor = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = torch.argmax(output, dim=0).byte().numpy()
    return (mask == 15).astype(np.uint8)


def extract_patch_using_deeplab(image, model):
    person_mask = predict_person_mask(image, model)
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    patch = image.crop((x, y, x + w, y + h))
    return (x, y, w, h), patch


def redact_patch(image, patch_coordinates, color=(0, 0, 0)):
    x, y, w, h = patch_coordinates
    redacted_image = image.copy()
    redacted_image.paste(color, (x, y, x + w, y + h))
    return redacted_image


def image_patch_to_bytes(patch):
    byte_array = bytearray()
    for y in range(patch.height):
        for x in range(patch.width):
            byte_array.extend(patch.getpixel((x, y)).to_bytes(3, 'big'))
    return bytes(byte_array)


def bytes_to_image_patch(byte_array, width, height):
    patch = Image.new('RGB', (width, height))
    idx = 0
    for y in range(height):
        for x in range(width):
            r, g, b = byte_array[idx:idx+3]
            patch.putpixel((x, y), (r, g, b))
            idx += 3
    return patch


def hide_patch(image_path, patch, x_offset, y_offset, output_image_path, encryption_key):
    image = Image.open(image_path).convert('RGB')
    patch_data = image_patch_to_bytes(patch)
    encrypted_patch_data = encrypt_data(patch_data, encryption_key)
    compressed_encrypted_patch_data = zlib.compress(encrypted_patch_data)
    encoded_patch_data = encode_data_with_ecc(compressed_encrypted_patch_data)
    hidden_image = hide_data_adaptive_steganography(image, encoded_patch_data)
    hidden_image.save(output_image_path)


def recover_patch(hidden_image_path, x_offset, y_offset, patch_width, patch_height, encryption_key):
    hidden_image = Image.open(hidden_image_path).convert('RGB')
    patch_data_length = patch_width * patch_height * 3
    extracted_encoded_patch_data = extract_data_adaptive_steganography(hidden_image, patch_data_length)
    decoded_compressed_encrypted_patch_data = decode_data_with_ecc(extracted_encoded_patch_data)
    decompressed_encrypted_patch_data = zlib.decompress(decoded_compressed_encrypted_patch_data)
    decrypted_patch_data = decrypt_data(decompressed_encrypted_patch_data, encryption_key)
    return bytes_to_image_patch(decrypted_patch_data, patch_width, patch_height)


def hide_and_redact_patch(image_path, output_image_path, model, encryption_key):
    image = Image.open(image_path).convert('RGB')
    patch_coordinates, patch = extract_patch_using_deeplab(image, model)
    if patch is None:
        print("No person found in the image.")
        return

    hide_patch(image_path, patch, *patch_coordinates, output_image_path, encryption_key)
    redacted_image = redact_patch(image, patch_coordinates)
    redacted_image.save("path/to/redacted_image.png")
    return patch_coordinates


def recover_and_restore_patch(hidden_image_path, redacted_image_path, patch_coordinates, encryption_key):
    recovered_patch = recover_patch(hidden_image_path, *patch_coordinates, encryption_key)
    redacted_image = Image.open(redacted_image_path).convert('RGB')
    redacted_image.paste(recovered_patch, (patch_coordinates[0], patch_coordinates[1]))
    redacted_image.save("path/to/restored_image.png")


def encode_data_with_ecc(data, ecc_symbols=10):
    rs = reedsolo.RSCodec(ecc_symbols)
    return rs.encode(data)


def decode_data_with_ecc(encoded_data, ecc_symbols=10):
    rs = reedsolo.RSCodec(ecc_symbols)
    try:
        decoded_data = rs.decode(encoded_data)
    except reedsolo.ReedSolomonError as e:
        print("Error: Unable to decode data due to too many errors.")
        decoded_data = None
    return decoded_data
 
def entropy(pixel_values):
    """Calculate the entropy of pixel values."""
    probs = [p / 255 for p in pixel_values]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def hide_data_adaptive_steganography(image, data):
    """Hide data in an image using adaptive steganography based on pixel entropy."""
    hidden_image = image.clone()
    bit_idx = 0
    for x in range(image.width):
        for y in range(image.height):
            pixel = tuple(image.getpixel((x, y)))
            if entropy(pixel) > 1.5 and bit_idx < len(data) * 8:
                byte_idx = bit_idx // 8
                data_bit = (data[byte_idx] >> (7 - (bit_idx % 8))) & 1
                new_pixel = (pixel[0] & 0xFE) | data_bit
                hidden_image.putpixel((x, y), new_pixel)
                bit_idx += 1
    return hidden_image


def extract_data_adaptive_steganography(image, data_length):
    """Extract data from an image using adaptive steganography based on pixel entropy."""
    extracted_data = bytearray()
    bit_idx = 0
    for x in range(image.width):
        for y in range(image.height):
            pixel = tuple(image.getpixel((x, y)))
            if entropy(pixel) > 1.5 and bit_idx < data_length * 8:
                data_bit = pixel[0] & 1
                if bit_idx % 8 == 0:
                    extracted_data.append(0)
                extracted_data[bit_idx // 8] |= data_bit << (7 - (bit_idx % 8))
                bit_idx += 1
    return bytes(extracted_data)


def main():
    parser = argparse.ArgumentParser(description="Steganography tool for hiding and recovering image patches")
    parser.add_argument("operation", choices=["hide", "recover"], help="Choose 'hide' or 'recover'")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to the output image")
    parser.add_argument("--redacted_image", help="Path to the redacted image (required for 'recover' operation)")
    parser.add_argument("--key", help="Path to the encryption key file")

    args = parser.parse_args()

    if args.key:
        with open(args.key, "rb") as key_file:
            encryption_key = key_file.read()
    else:
        encryption_key = generate_encryption_key()
        with open("key.bin", "wb") as key_file:
            key_file.write(encryption_key)
        print("Generated encryption key saved as 'key.bin'")

    model = load_deeplab_model()

    if args.operation == "hide":
        hide_and_redact_patch(args.input_image, args.output_image, model, encryption_key)
        print("Person patch hidden and image redacted.")
    elif args.operation == "recover":
        if not args.redacted_image:
            print("Error: --redacted_image is required for 'recover' operation.")
            sys.exit(1)

        patch_coordinates = input("Enter patch coordinates as 'x y w h': ").split()
        patch_coordinates = tuple(map(int, patch_coordinates))

        recover_and_restore_patch(args.input_image, args.redacted_image, patch_coordinates, encryption_key)
        print("Person patch recovered and image restored.")


if __name__ == "__main__":
    main()