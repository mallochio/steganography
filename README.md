## 💔 Work in progress 

# Image Steganography

This repository contains implementations of various image steganography techniques. 

## Techniques Implemented

1. **LSB (Least Significant Bit) Steganography** - Hides a secret image in the least significant bit of each pixel of a cover image.
2. **DCT (Discrete Cosine Transform) Steganography** - Hides information in the frequency domain of an image.

## Usage

### LSB Steganography

```python
from src.steganography import lsb_encode, lsb_decode
from PIL import Image

# Load images
cover_img = Image.open('images/person_mask.jpg')
secret_img = Image.open('images/me.png').convert('1')  # Convert to binary

# Encode
encoded_img = lsb_encode(cover_img, secret_img)
encoded_img.save('encoded_lsb.png')

# Decode
decoded_img = lsb_decode(encoded_img)
decoded_img.save('decoded_secret.png')
```

### DCT Steganography

```python
import cv2
from src.steganography import dct_encode, dct_decode
import numpy as np

# Load images
cover_img = cv2.imread('images/person_mask.jpg', cv2.IMREAD_GRAYSCALE)
secret_img = cv2.imread('images/me.png', cv2.IMREAD_GRAYSCALE)

# Encode
stego_img = dct_encode(cover_img, secret_img, scale_factor=0.1)
cv2.imwrite('encoded_dct.png', stego_img)

# Decode
decoded_img = dct_decode(stego_img, cover_img, scale_factor=0.1, secret_shape=secret_img.shape)
cv2.imwrite('decoded_secret.png', decoded_img)
```

## License

MIT
## Future Work

- Implement wavelet-based steganography
- Add encryption support
