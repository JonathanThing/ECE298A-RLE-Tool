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
    # Create an empty image array with 3 channels (RGB)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y, row_rle in enumerate(frame_rle):
        if row_rle is None:
            continue
        x = 0
        for color, length in row_rle:
            # Unquantize the color
            r = int((color >> 5) & 0x07) * 255 // 7  # Extract 3-bit red and scale to 0-255
            g = int((color >> 2) & 0x07) * 255 // 7  # Extract 3-bit green and scale to 0-255
            b = int((color & 0x03)) * 255 // 3       # Extract 2-bit blue and scale to 0-255

            # Fill the pixels with the unquantized color
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

# each strip must be at least 3 pixels
# every 6 consecutive strips in one row must sum to at least 36 pixels
# warn if the sum of pixels in a row is not 640
def verify_rle_row(row_rle):
    total_length = 0
    window_sum = 0
    for i in range(len(row_rle)):
        if row_rle[i][1] < 3:
            print("RLE strip too short:")
            return False

        total_length += row_rle[i][1]
        window_sum += row_rle[i][1]

        # Check every 6-strip window
        if i >= 5:
            if window_sum < 36:
                print("RLE window too short:")
                return False
            window_sum -= row_rle[i-5][1]

    if total_length != 640:
        print("RLE row length incorrect:")
        return False

    return True

def verify_image(image_path):
    from PIL import Image
    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels = np.array(img)
    height, width = pixels.shape[:2]
    quantized_pixels = rgb_to_8bit(pixels, bw_mode=False)
    all_rows_valid = True
    error_found = False
    error_row = None
    error_run_idx = None
    error_type = None
    error_run_start = None
    error_run_end = None
    for y in range(height):
        runs = []
        run_length = 1
        current_color = quantized_pixels[y, 0]
        for x in range(1, width):
            if quantized_pixels[y, x] == current_color:
                run_length += 1
            else:
                runs.append(run_length)
                current_color = quantized_pixels[y, x]
                run_length = 1
        runs.append(run_length)
        valid = True
        for i, length in enumerate(runs):
            start = sum(runs[:i])
            end = start + length
            if length < 3:
                print(f"Row {y}: strip too short at x={start}, length {length}, pixel range {start}-{end-1}")
                if not error_found:
                    error_found = True
                    error_row = y
                    error_run_idx = i
                    error_type = 'strip'
                    error_run_start = start
                    error_run_end = end
                valid = False
        for i in range(len(runs) - 5):
            window_sum = sum(runs[i:i+6])
            window_start = sum(runs[:i])
            window_end = sum(runs[:i+6])
            if window_sum < 36:
                print(f"Row {y}: window too short at x={window_start}, sum {window_sum}, pixel range {window_start}-{window_end-1}")
                if not error_found:
                    error_found = True
                    error_row = y
                    error_run_idx = i
                    error_type = 'window'
                valid = False
        total_length = sum(runs)
        if total_length != width:
            print(f"Row {y}: row length incorrect, got {total_length}, expected {width}")
            if not error_found:
                error_found = True
                error_row = y
                error_run_idx = None
                error_type = 'rowlen'
            valid = False
        if not valid:
            all_rows_valid = False
    return all_rows_valid
    

import sys

# if main
if __name__ == "__main__":
    #read args to get image to verify rle
    if len(sys.argv) < 2:
        print("Usage: python utils.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if (verify_image(image_path)):
        print("RLE verification passed.")
    else:
        print("RLE verification failed.")