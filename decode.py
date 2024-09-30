import cv2
from tqdm import tqdm
import magic
import os
import time
import numpy as np

file_path = r"output_video.avi"

pixel_size = 2
pixel = pixel_size ** 2

cap = cv2.VideoCapture(file_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rgb_values = []

def bin2bin(lst):
    binary_string = ''.join(str(bit) for bit in lst)
    byte_array = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
    return byte_array

for i in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

    for y in tqdm(range(int(height / pixel_size))):
        for x in range(int(width / pixel_size)):
            region = frame[y*pixel_size:y*pixel_size+pixel_size, x*pixel_size:x*pixel_size+pixel_size, :]
            total_rgb = np.sum(region) / (3 * pixel)
            rgb_values.append(round(total_rgb / 255))

cap.release()

while rgb_values and rgb_values[-1] == 0: rgb_values.pop()
file_bin = bin2bin(rgb_values)

with open("output.bin" , 'wb') as f:
    f.write(file_bin)

try:
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file("output.bin")
    extension = mime_type.split('/')[-1]
    os.rename("output.bin", f"output.{extension}")
    print(f"decoded: output.{extension}")
except Exception as e:
    print(e)