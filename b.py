import struct
from PIL import Image
import numpy as np
import os

AUDIO_INSTRUCTION = 0x3FF00
STOP_INSTRUCTION = 0x30000

END_OF_LINE_MARKER = b'EOL'

def rgb_to_8bit(r, g, b, threshold=128):
    # Convert to grayscale using luminance formula
    luminance = int(0.299 * r + 0.587 * g + 0.114 * b)
    return 0xFF if luminance >= threshold else 0x00

def pixel_instruction(run_length, colour):
    rle_instruction = np.uint32((run_length << 8)) | colour
    rle_bytes = struct.pack('>I', rle_instruction)[1:]
    return rle_bytes

def audio_instruction(audio_sample):
    audio_instruction = AUDIO_INSTRUCTION | audio_sample
    audio_bytes = struct.pack('>I', audio_instruction)[1:]
    return audio_bytes

def stop_instruction():
    stop_instruction = STOP_INSTRUCTION
    stop_bytes = struct.pack('>I', stop_instruction)[1:]
    return stop_bytes

def enforce_rle_constraints(rle_strips):
    for i in range(len(rle_strips)):
        if rle_strips[i]['length'] < 3:
            rle_strips[i]['length'] = 3
    for group_start in range(0, len(rle_strips), 6):
        group_end = min(group_start + 6, len(rle_strips))
        group = rle_strips[group_start:group_end]
        total_length = sum(strip['length'] for strip in group)
        if total_length < 36:
            deficit = 36 - total_length
            pixels_per_strip = deficit // len(group)
            remainder = deficit % len(group)
            for i, strip in enumerate(group):
                strip['length'] += pixels_per_strip
                if i < remainder:
                    strip['length'] += 1
    return rle_strips

def create_modified_image(original_pixels, rle_strips, width, height):
    new_pixels = np.zeros_like(original_pixels)
    for row in range(height):
        row_strips = [strip for strip in rle_strips if strip['row'] == row]
        pixel_pos = 0
        for strip in row_strips:
            end_pos = min(pixel_pos + strip['length'], width)
            color_8bit = strip['color']
            if color_8bit == 0xFF:
                r, g, b = 255, 255, 255
            elif color_8bit == 0x00:
                r, g, b = 0, 0, 0
            else:
                r = ((color_8bit >> 5) & 0x07) * 255 // 7
                g = ((color_8bit >> 2) & 0x07) * 255 // 7
                b = (color_8bit & 0x03) * 255 // 3
            for col in range(pixel_pos, end_pos):
                if col < width:
                    new_pixels[row, col] = [r, g, b]
            pixel_pos = end_pos
            if pixel_pos >= width:
                break
    return new_pixels

def process_all_frames(frames_folder="frames/", output_folder="frames_new/"):

    # Setup output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for frame_file in os.listdir(output_folder):
        if frame_file.endswith(".png"):
            os.remove(os.path.join(output_folder, frame_file))

    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    for frame_file in frame_files:
        input_path = os.path.join(frames_folder, frame_file)
        output_path = os.path.join(output_folder, frame_file)
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = np.array(img)
        height, width = pixels.shape[:2]
        rle_strips = []
        for i in range(height):
            run_length = 0
            previous_colour = -1
            for j in range(width):
                r, g, b = pixels[i][j]
                current_colour = rgb_to_8bit(r, g, b)
                if previous_colour == -1:
                    previous_colour = current_colour
                elif previous_colour != current_colour:
                    rle_strips.append({
                        'row': i,
                        'length': run_length,
                        'color': previous_colour
                    })
                    previous_colour = current_colour
                    run_length = 0
                run_length += 1
            rle_strips.append({
                'row': i,
                'length': run_length,
                'color': previous_colour
            })
        rle_strips = enforce_rle_constraints(rle_strips)
        modified_pixels = create_modified_image(pixels, rle_strips, width, height)
        modified_img = Image.fromarray(modified_pixels.astype(np.uint8), 'RGB')
        modified_img.save(output_path)
        print(f"Processed {frame_file} -> {output_path}")

if __name__ == "__main__":
    process_all_frames()