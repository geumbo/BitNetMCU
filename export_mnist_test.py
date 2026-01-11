"""
Export MNIST test set to binary file for rv32emu testing.

Output format: [label 1B][image 256B] × 10000
- label: uint8_t (0-9)
- image: int8_t[256] (16×16 pixels, normalized to [-128, 127])

Total file size: 10000 × (1 + 256) = 2,570,000 bytes (~2.45 MB)
"""

import struct
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms

# MNIST preprocessing (same as test_inference.py)
IMAGE_SIZE = 16

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
])

# Load MNIST test set
test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

print(f"MNIST test set loaded: {len(test_data)} images")

# Export to binary file
output_path = Path('mnist_test.bin')

with open(output_path, 'wb') as f:
    for i, (image, label) in enumerate(test_data):
        # Convert to numpy and flatten
        input_data = image.numpy().flatten()
        
        # Dynamic scaling per image (same as test_inference.py)
        scale = 127.0 / max(np.abs(input_data).max(), 1e-5)
        scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.int8)
        
        # Write: [label 1B][image 256B]
        f.write(struct.pack('B', label))
        f.write(scaled_data.tobytes())
        
        if (i + 1) % 1000 == 0:
            print(f"Exported {i + 1} / {len(test_data)} images")

file_size = output_path.stat().st_size
print(f"\nExported to: {output_path}")
print(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
print(f"Expected: {len(test_data) * 257} bytes")
