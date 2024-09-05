#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   visual_encryption.py
@Time    :   2023/07/10 08:48:37
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Implementation of a visual encryption algorithm where an image is split into two shares and the original image can be recovered by stacking the two shares on top of each other.
"""

# Importing libraries
import cv2
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def get_args():
    """
    Function to get the arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output folder."
    )
    parser.add_argument(
        "--share1", type=str, required=True, help="Path to the output share 1 image."
    )
    parser.add_argument(
        "--share2", type=str, required=True, help="Path to the output share 2 image."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Threshold value for the visual encryption algorithm.",
    )

    return parser.parse_args()


def visual_encryption(input_image, output_folder, share1, share2, threshold):
    """
    Function to implement the visual encryption algorithm.
    """
    # Reading the input image
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = np.array(image)

    share_1 = _extracted_from_visual_encryption_12()
    share_2 = _extracted_from_visual_encryption_12()
    # Performing the visual encryption
    for i in tqdm(range(512)):
        for j in range(512):
            if image[i][j][0] > threshold:
                share_1[i][j] = [0, 0, 0]
                share_2[i][j] = [0, 0, 0]
            else:
                share_1[i][j] = [255, 255, 255]
                share_2[i][j] = [255, 255, 255]

    # Saving the share 1 image
    share_1 = Image.fromarray(share_1)
    share_1.save(share1)

    # Saving the share 2 image
    share_2 = Image.fromarray(share_2)
    share_2.save(share2)

    output_image = _extracted_from_visual_encryption_12()
    output_image = Image.fromarray(output_image)
    output_image.save(f"{output_folder}/output_image.png")


# TODO Rename this here and in `visual_encryption`
def _extracted_from_visual_encryption_12():
    # Creating the share image
    result = np.zeros((512, 512, 3), dtype=np.uint8)
    result = np.array(result)
    result = result + 255
    return result


def main():
    """
    Main function.
    """
    args = get_args()
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    visual_encryption(
        args.input_image, args.output_folder, args.share1, args.share2, args.threshold
    )


if __name__ == "__main__":
    main()
