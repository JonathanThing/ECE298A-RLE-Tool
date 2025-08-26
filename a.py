import struct
import time
from PIL import Image
import numpy as np
import os
import concurrent.futures
from multiprocessing import cpu_count

AUDIO_INSTRUCTION = 0x3FF00
STOP_INSTRUCTION = 0x30000

GROUP_SIZE = 6
MIN_GROUP_LENGTH = 36

# Converts 8 bit per pixel RGB to RRRGGGBB format (8 bits in total) - Vectorized version
def rgb_to_8bit(pixels, bw_mode=False, threshold=128):
    """Vectorized version of rgb_to_8bit for faster processing"""
    if bw_mode:
        luminance = np.dot(pixels, [0.299, 0.587, 0.114])
        return np.where(luminance >= threshold, 0xFF, 0x00).astype(np.uint8)
    else:
        r_3bit = np.round(pixels[:, :, 0] / 255 * 7).astype(np.uint8)
        g_3bit = np.round(pixels[:, :, 1] / 255 * 7).astype(np.uint8)
        b_2bit = np.round(pixels[:, :, 2] / 255 * 3).astype(np.uint8)
        return (r_3bit << 5) | (g_3bit << 2) | b_2bit

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
    i = 0
    while i < len(rle_strips):
        if rle_strips[i]['length'] == 2: # Add a pixel to current, remove one from another
            rle_strips[i]['length'] = 3
            left = rle_strips[i-1]['length'] if i > 0 else 0
            right = rle_strips[i+1]['length'] if i+1 < len(rle_strips) else 0

            if left > right and left > 3:
                rle_strips[i-1]['length'] -= 1
            elif right != 0:
                rle_strips[i+1]['length'] -= 1
        elif rle_strips[i]['length'] < 2: # Remove the strip

            if rle_strips[i]['length'] == 1: # If has one pixel, move it to the smallest of its neighbor
                left = rle_strips[i-1]['length'] if i > 0 else 999
                right = rle_strips[i+1]['length'] if i+1 < len(rle_strips) else 999
                if left < right:
                    rle_strips[i-1]['length'] += 1
                else:
                    rle_strips[i+1]['length'] += 1 
            del rle_strips[i]
        i += 1

    for group_start in range(0, len(rle_strips), GROUP_SIZE):
        group_end = min(group_start + GROUP_SIZE, len(rle_strips))
        group = rle_strips[group_start:group_end]
        total_length = sum(strip['length'] for strip in group)
        if total_length < MIN_GROUP_LENGTH:
            deficit = MIN_GROUP_LENGTH - total_length
            # Sort strips by length descending
            sorted_indices = sorted(range(len(group)), key=lambda i: -group[i]['length'])
            for idx in sorted_indices:
                if deficit == 0:
                    break
                group[idx]['length'] += 1
                deficit -= 1
    return rle_strips

def create_modified_image_optimized(original_pixels, rle_strips, width, height):
    """Optimized version using numpy array operations"""
    new_pixels = np.zeros_like(original_pixels)
    for row in range(height):
        row_strips = rle_strips[row]
        pixel_pos = 0
        for strip in row_strips:
            end_pos = min(pixel_pos + strip['length'], width)
            color_8bit = strip['color']
            
            # Decode color once
            if color_8bit == 0xFF:
                color = [255, 255, 255]
            elif color_8bit == 0x00:
                color = [0, 0, 0]
            else:
                r = round(int((color_8bit >> 5) & 0x07) * 255 / 7)
                g = round(int((color_8bit >> 2) & 0x07) * 255 / 7)
                b = round(int((color_8bit & 0x03)) * 255 / 3)
                color = [r, g, b]

            # Use vectorized assignment instead of loop
            if end_pos > pixel_pos and pixel_pos < width:
                actual_end = min(end_pos, width)
                new_pixels[row, pixel_pos:actual_end] = color
            
            pixel_pos = end_pos
            if pixel_pos >= width:
                break
    return new_pixels

def process_frame(input_path, output_path, bw_mode):
    """Process a single frame - extracted for parallel processing"""
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    pixels = np.array(img)
    height, width = pixels.shape[:2]
    
    # Use vectorized quantization for faster processing
    quantized = rgb_to_8bit(pixels, bw_mode)
    
    rle_strips = []
    for i in range(height):
        row_strips = []
        run_length = 1
        previous_colour = quantized[i, 0]
        
        for j in range(1, width):
            current_colour = quantized[i, j]
            if previous_colour != current_colour:
                row_strips.append({
                    'row': i,
                    'length': run_length,
                    'color': previous_colour
                })
                previous_colour = current_colour
                run_length = 1
            else:
                run_length += 1
        
        # Add final run
        row_strips.append({
            'row': i,
            'length': run_length,
            'color': previous_colour
        })
        rle_strips.append(row_strips)

    rle_strips = [enforce_rle_constraints(row) for row in rle_strips]
    modified_pixels = create_modified_image_optimized(pixels, rle_strips, width, height)
    modified_img = Image.fromarray(modified_pixels.astype(np.uint8), 'RGB')
    modified_img.save(output_path)
    return input_path

def process_frame_to_rle(input_path, bw_mode):
    """Process a single frame and return RLE data instead of saving image"""
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    pixels = np.array(img)
    height, width = pixels.shape[:2]
    
    # Use vectorized quantization for faster processing
    quantized = rgb_to_8bit(pixels, bw_mode)
    
    rle_strips = []
    for i in range(height):
        row_strips = []
        run_length = 1
        previous_colour = quantized[i, 0]
        
        for j in range(1, width):
            current_colour = quantized[i, j]
            if previous_colour != current_colour:
                row_strips.append({
                    'row': i,
                    'length': run_length,
                    'color': previous_colour
                })
                previous_colour = current_colour
                run_length = 1
            else:
                run_length += 1
        
        # Add final run
        row_strips.append({
            'row': i,
            'length': run_length,
            'color': previous_colour
        })
        rle_strips.append(row_strips)

    rle_strips = [enforce_rle_constraints(row) for row in rle_strips]
    return rle_strips

def create_rle_binary(frames_folder="frames_modified/", output_file="rle_data.bin", bw_mode=False, max_workers=None):
    """Convert frames to RLE binary format"""
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    
    if max_workers is None:
        max_workers = min(cpu_count(), len(frame_files))
    
    print(f"Converting {len(frame_files)} frames to RLE binary using {max_workers} workers...")
    
    # Process frames in parallel to get RLE data
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for frame_file in frame_files:
            input_path = os.path.join(frames_folder, frame_file)
            futures.append(executor.submit(process_frame_to_rle, input_path, bw_mode))
        
        # Collect results in order
        all_rle_data = []
        completed = 0
        for future in futures:
            try:
                rle_data = future.result()
                all_rle_data.append(rle_data)
                completed += 1
                if completed % 10 == 0 or completed == len(frame_files):
                    print(f"Processed {completed}/{len(frame_files)} frames")
            except Exception as e:
                print(f"Error processing frame: {e}")
                all_rle_data.append([])  # Add empty data for failed frames

    # Write binary data
    print(f"Writing RLE data to {output_file}...")
    with open(output_file, 'wb') as f:
        for frame_idx, frame_rle in enumerate(all_rle_data):
            # Write frame header (frame number)
            f.write(struct.pack('>I', frame_idx))
            
            for row_strips in frame_rle:
                for strip in row_strips:
                    # Write RLE instruction for each strip
                    rle_bytes = pixel_instruction(strip['length'], strip['color'])
                    f.write(rle_bytes)
                
            # Write stop instruction at end of frame
            f.write(stop_instruction())
    
    print(f"RLE binary file created: {output_file}")

def process_all_frames(frames_folder="frames/", output_folder="frames_modified/", bw_mode=False, max_workers=None):
    """Optimized parallel frame processing"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Clear output folder
    for frame_file in os.listdir(output_folder):
        if frame_file.endswith(".png"):
            os.remove(os.path.join(output_folder, frame_file))

    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    
    if max_workers is None:
        max_workers = min(cpu_count(), len(frame_files))
    
    print(f"Processing {len(frame_files)} frames using {max_workers} workers...")
    
    # Prepare tasks
    tasks = []
    for frame_file in frame_files:
        input_path = os.path.join(frames_folder, frame_file)
        output_path = os.path.join(output_folder, frame_file)
        tasks.append((input_path, output_path, bw_mode))
    
    # Process frames in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, *task) for task in tasks]
        
        # Show progress
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(frame_files):
                    print(f"Processed {completed}/{len(frame_files)} frames")
            except Exception as e:
                print(f"Error processing frame: {e}")

if __name__ == "__main__":
    start_time = time.time()

    print("Select quantization mode:")
    print("1: Normal quantization")
    print("2: Black and white mode")
    quant_mode = input("Enter 1 or 2: ").strip()
    bw_mode = (quant_mode == "2")
    
    print("\nSelect output mode:")
    print("1: Generate modified frame images")
    print("2: Generate RLE binary file")
    output_mode = input("Enter 1 or 2: ").strip()
    
    if output_mode == "2":
        create_rle_binary(bw_mode=bw_mode)
    else:
        process_all_frames(bw_mode=bw_mode)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")