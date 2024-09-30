import cv2
import numpy as np
import asyncio
import concurrent.futures
from tqdm import tqdm

file_path = "input.png"
width = 1280
height = 720
pixel_size = 2

async def read_file(file_path):
    loop = asyncio.get_event_loop()
    with open(file_path, 'br') as f:
        data = await loop.run_in_executor(None, f.read)  # 非同期でファイルを読み込む
    return data

async def encode_data(data):
    loop = asyncio.get_event_loop()
    encoded_file = await loop.run_in_executor(None, lambda: [bit for byte in data for bit in format(byte, '08b')])
    encoded_file = list(map(int, encoded_file))
    return encoded_file

async def process_chunk(chunk, width, height, pixel_size):
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, lambda: np.array(chunk, dtype=np.uint8).reshape((height, width)))
    frame = np.repeat(np.repeat(frame, pixel_size, axis=0), pixel_size, axis=1)
    return frame

async def list_to_frames(pixel_list, width=320, height=180, pixel_size=1):
    total_pixels = width * height
    frames = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(0, len(pixel_list), total_pixels)):
            chunk = list(map(lambda x: x * 255, pixel_list[i:i + total_pixels]))
            if len(chunk) < total_pixels:
                chunk += [0] * (total_pixels - len(chunk))
            future = asyncio.ensure_future(process_chunk(chunk, width, height, pixel_size))
            futures.append(future)

        for future in tqdm(asyncio.as_completed(futures)):
            frame = await future
            frames.append(frame)
    
    return frames

async def main():
    # ファイルを非同期で読み込む
    data = await read_file(file_path)
    
    # データを非同期でエンコードする
    encoded_file = await encode_data(data)
    
    frames = await list_to_frames(encoded_file, int(width / pixel_size), int(height / pixel_size), pixel_size)

    video_name = 'output_video.avi'
    fps = 30
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    video = cv2.VideoWriter(video_name, fourcc, fps, frame_size, isColor=False)

    for frame in tqdm(frames):
        video.write(frame)

    video.release()
    print(f"{video_name} の作成が完了しました。")
    print(len(encoded_file))

# 非同期関数を実行
asyncio.run(main())
