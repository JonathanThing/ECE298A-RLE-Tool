import struct
import numpy as np
from multiprocessing import cpu_count
import constants

# Converts 8 bit per colour to RRRGGGBB format (8 bits for one pixel)
def rgb_to_8bit(pixels, bw_mode=False, threshold=128):
    if bw_mode:
        luminance = np.dot(pixels, [0.299, 0.587, 0.114])
        return np.where(luminance >= threshold, 0xFF, 0x00).astype(np.uint8)
    else:
        r_3bit = (pixels[:, :, 0] >> 5) & 0x07
        g_3bit = (pixels[:, :, 1] >> 5) & 0x07
        b_2bit = (pixels[:, :, 2] >> 6) & 0x03
        return (r_3bit << 5) | (g_3bit << 2) | b_2bit
    
def rle_to_rgb(frame_rle, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y, row_rle in enumerate(frame_rle):
        if row_rle is None:
            continue
        x = 0
        for colour, length in row_rle:
            r = int((colour >> 5) & 0x07) * 255 // 7  # 3-bit
            g = int((colour >> 2) & 0x07) * 255 // 7  # 3-bit
            b = int((colour & 0x03)) * 255 // 3       # 2-bit

            img[y, x:x+length] = [r, g, b]
            x += length
    
    return img

def pixel_instruction(run_length, colour):
    rle_instruction = np.uint32((run_length << 8)) | colour
    rle_bytes = struct.pack('>I', rle_instruction)[1:]
    return rle_bytes

def audio_instruction(audio_sample):
    audio_instruction = constants.AUDIO_INSTRUCTION | audio_sample
    audio_bytes = struct.pack('>I', audio_instruction)[1:]
    return audio_bytes

def stop_instruction():
    stop_instruction = constants.STOP_INSTRUCTION
    stop_bytes = struct.pack('>I', stop_instruction)[1:]
    return stop_bytes

# Each strip must be at least 3 pixels
# Every 6 consecutive strips in one row must sum to at least 36 pixels
# Entire row must be 640 pixels
def verify_rle_row(row_rle):
    total_length = 0
    window_sum = 0
    
    for i in range(len(row_rle)):
        if row_rle[i][1] < 3:
            print("RLE strip too short:")
            return False

        total_length += row_rle[i][1]
        window_sum += row_rle[i][1]

        if i >= 5:
            if window_sum < 36:
                print("RLE window too short:")
                return False
            window_sum -= row_rle[i-5][1]

    if total_length != 640:
        print("RLE row length incorrect:")
        return False

    return True