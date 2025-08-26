import struct
from PIL import Image
import numpy as np
import os
import concurrent.futures
from multiprocessing import cpu_count

import constants
import utils

# Global similarity threshold for merging strips
SIMILARITY_THRESHOLD = 0.7  # User can configure this value

def verify_rle(frame_rle):
    for row in frame_rle:
        if not utils.verify_rle(row):
            print("RLE verification failed for row:", row)
            return False
    return True

def set_rle_length(row_rle, index, new_length):
    color = row_rle[index][0]
    row_rle[index] = (color, new_length)

def increment_rle_length(row_rle, index, increment):
    color, current_length = row_rle[index]
    row_rle[index] = (color, current_length + increment)

def enforce_rle_constraints(row_rle, bw_mode=False):
    if bw_mode:
        # TODO: Implement logic for black-and-white mode
        pass
    else:  # Handle color mode
        # Step 1: Remove artifacts (strips < 3 pixels) by absorbing into neighbors
        row_rle = remove_short_strips(row_rle)
        
        # Step 2: Enforce group constraints (every 6 strips sum to >= 36)
        row_rle = enforce_group_constraints(row_rle)
        
        # Note: We don't trim to maintain total pixel count
        # The borrowing system in extend_window_strips maintains conservation
        
    return row_rle

def remove_short_strips(row_rle):
    """Remove strips that are less than 3 pixels long by absorbing them into neighbors"""
    i = len(row_rle) - 1
    while i >= 0:
        color, length = row_rle[i]
        if length < 3:
            # If at the rightmost strip, only merge left
            if i == len(row_rle) - 1:
                increment_rle_length(row_rle, i-1, length)
                del row_rle[i]
                i -= 1
                continue
            # If at the leftmost strip, only merge right
            elif i == 0:
                increment_rle_length(row_rle, i+1, length)
                del row_rle[i]
                # No need to decrement i, as loop will exit
                continue
            # Otherwise, use normal logic
            absorb_strip_into_neighbor(row_rle, i)
            i -= 1
            continue
        i -= 1
    return row_rle

def absorb_strip_into_neighbor(row_rle, index):
    """Absorb a short strip into the most appropriate neighbor"""
    color, length = row_rle[index]
    
    left_available = index > 0
    right_available = index < len(row_rle) - 1
    
    if left_available and right_available:
        left_color, left_length = row_rle[index-1]
        right_color, right_length = row_rle[index+1]
        
        # Priority 1: Merge with same color neighbor
        if left_color == color:
            increment_rle_length(row_rle, index-1, length)
        elif right_color == color:
            increment_rle_length(row_rle, index+1, length)
        else:
            # Priority 2: Choose neighbor with similar color (minimal visual impact)
            left_similarity = calculate_color_similarity(color, left_color)
            right_similarity = calculate_color_similarity(color, right_color)
            
            # Only merge if similarity is high enough to avoid distortion
            if left_similarity >= SIMILARITY_THRESHOLD and left_similarity >= right_similarity:
                increment_rle_length(row_rle, index-1, length)
            elif right_similarity >= SIMILARITY_THRESHOLD:
                increment_rle_length(row_rle, index+1, length)
            else:
                # If neither neighbor is similar enough, merge with the longer one
                # to minimize visual impact
                if left_length >= right_length:
                    increment_rle_length(row_rle, index-1, length)
                else:
                    increment_rle_length(row_rle, index+1, length)
                
    elif left_available:
        increment_rle_length(row_rle, index-1, length)
    elif right_available:
        increment_rle_length(row_rle, index+1, length)
    else:
        # This shouldn't happen in normal cases, but if it does, keep the strip
        return
    
    del row_rle[index]

def calculate_color_similarity(color1, color2):
    """Simple similarity based on how many bits differ"""
    if color1 == color2:
        return 1.0
    
    # Count differing bits
    xor_result = color1 ^ color2
    differing_bits = bin(xor_result).count('1')
    
    # Normalize by total bits (8)
    similarity = 1.0 - (differing_bits / 8.0)
    
    return similarity

def enforce_group_constraints(row_rle):
    """Ensure every group of 6 consecutive strips sums to at least 36 pixels"""
    i = 0
    max_start = max(0, len(row_rle) - constants.GROUP_SIZE)
    while i <= max_start:
        window = row_rle[i:i+constants.GROUP_SIZE]
        total = sum([strip[1] for strip in window])
        if total < constants.MIN_GROUP_LENGTH:
            deficit = constants.MIN_GROUP_LENGTH - total
            # Strategy 1: Try to merge similar colored strips in the window
            if try_merge_in_window(row_rle, i, deficit):
                # Window may have changed, so recalculate max_start
                max_start = max(0, len(row_rle) - constants.GROUP_SIZE)
                continue  # Window changed, re-evaluate
            # Strategy 2: Extend strips intelligently (distribute deficit)
            extend_window_strips(row_rle, i, deficit)
        i += 1
    return row_rle

def try_merge_in_window(row_rle, start_index, deficit):
    """Try to merge strips with same color in the current window"""
    end_index = start_index + constants.GROUP_SIZE
    
    for i in range(start_index, min(end_index - 1, len(row_rle) - 1)):
        for j in range(i + 1, min(end_index, len(row_rle))):
            if row_rle[i][0] == row_rle[j][0]:
                # Found same colors, check if merging helps
                # Calculate cost of merging (pixels between that will change color)
                merge_cost = sum(row_rle[k][1] for k in range(i + 1, j))
                
                if merge_cost <= deficit:
                    # Merge is worthwhile
                    merge_strips_range(row_rle, i, j)
                    return True
    
    return False

def merge_strips_range(row_rle, start, end):
    """Merge strips from start to end index (inclusive)"""
    total_length = sum(row_rle[k][1] for k in range(start, end + 1))
    color = row_rle[start][0]  # Use the first strip's color
    
    # Remove all strips in range and replace with merged strip
    for _ in range(end - start):
        del row_rle[start + 1]
    
    row_rle[start] = (color, total_length)

def extend_window_strips(row_rle, start_index, deficit):
    """Extend strips in the window to meet minimum group length by borrowing from strips outside the window"""
    end_index = min(start_index + constants.GROUP_SIZE, len(row_rle))
    
    # Find strips outside the current window that can donate pixels
    donor_candidates = []
    for i in range(len(row_rle)):
        if i < start_index or i >= end_index:  # Outside current window
            if row_rle[i][1] > 3:  # Can afford to give up pixels
                donor_candidates.append((i, row_rle[i][1]))
    
    # Sort donors by how much they can give (longest first)
    donor_candidates.sort(key=lambda x: x[1], reverse=True)
    
    remaining_deficit = deficit
    
    # Try to borrow from donors outside the window
    for donor_index, donor_length in donor_candidates:
        if remaining_deficit <= 0:
            break
        
        # How much can this donor give without violating constraints?
        max_donation = donor_length - 3  # Must keep at least 3
        
        # Check if donor is part of a window that would become invalid
        donor_safe = True
        for window_start in range(max(0, donor_index - constants.GROUP_SIZE + 1), 
                                 min(len(row_rle) - constants.GROUP_SIZE + 1, donor_index + 1)):
            window_sum = sum(row_rle[j][1] for j in range(window_start, window_start + constants.GROUP_SIZE))
            if window_sum - max_donation < constants.MIN_GROUP_LENGTH:
                max_donation = window_sum - constants.MIN_GROUP_LENGTH
                if max_donation <= 0:
                    donor_safe = False
                    break
        
        if not donor_safe or max_donation <= 0:
            continue
        
        # Take what we need, up to what the donor can give
        donation = min(remaining_deficit, max_donation)
        
        # Remove from donor
        increment_rle_length(row_rle, donor_index, -donation)
        
        # Distribute to strips in the current window
        strips_in_window = [(i, row_rle[i][1]) for i in range(start_index, end_index)]
        strips_in_window.sort(key=lambda x: x[1], reverse=True)  # Longest first
        
        per_strip = max(1, donation // len(strips_in_window))
        remaining_donation = donation
        
        for strip_index, current_length in strips_in_window:
            if remaining_donation <= 0:
                break
            
            add_amount = min(per_strip, remaining_donation)
            increment_rle_length(row_rle, strip_index, add_amount)
            remaining_donation -= add_amount
        
        # If there's still donation left, add to the longest strips
        while remaining_donation > 0:
            longest_index = max(range(start_index, end_index), key=lambda i: row_rle[i][1])
            increment_rle_length(row_rle, longest_index, 1)
            remaining_donation -= 1
        
        remaining_deficit -= donation
    
    # If we still have deficit and couldn't borrow enough, we have to violate the total constraint
    # This should be rare if the original image has proper pixel count


# def create_modified_image_optimized(original_pixels, rle_strips, width, height):
    
def process_frame(input_path, output_path, bw_mode):
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    pixels = np.array(img)
    height = pixels.shape[0]
    width = pixels.shape[1]

    pixels = utils.rgb_to_8bit(pixels, bw_mode)

    frame_rle = []
    for i in range(height):
        row_rle = []
        run_length = 1
        previous_colour = pixels[i, 0]
        for j in range(1, width):
            current_colour = pixels[i, j]
            if current_colour == previous_colour:
                run_length += 1
            else:
                row_rle.append((previous_colour, run_length))
                previous_colour = current_colour
                run_length = 1
        row_rle.append((previous_colour, run_length))
        row_rle = enforce_rle_constraints(row_rle)
        frame_rle.append(row_rle)

    #create modified image
    modified_pixels = utils.rle_to_rgb(frame_rle, width, height)
    modified_img = Image.fromarray(modified_pixels)
    modified_img.save(output_path)

def process_all_frames_concurrent(bw_mode=False, max_workers=None):
    """Process all frames concurrently using multiprocessing"""
    input_folder = constants.FRAMES_FOLDER
    output_folder = constants.MODIFIED_FRAMES_FOLDER
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.exists(output_folder):
        confirm = input(f"Output folder {output_folder} already exists. Do you want to delete it? (y/n): ")
        if confirm.lower() == 'y':
            for frame_file in os.listdir(output_folder):
                if frame_file.endswith(".png"):
                    os.remove(os.path.join(output_folder, frame_file))
        else:
            print("Exiting without processing frames.")
            return

    # Get list of all frame files
    frame_files = []
    if os.path.exists(input_folder):
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_files.append(filename)
    
    if not frame_files:
        print(f"No frame files found in {input_folder}")
        return
    
    print(f"Found {len(frame_files)} frames to process")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(frame_files))
    
    print(f"Using {max_workers} workers for processing")
    
    # Create list of tasks (input_path, output_path, bw_mode)
    tasks = []
    for filename in frame_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        tasks.append((input_path, output_path, bw_mode))
    
    # Process frames concurrently
    processed_count = 0
    failed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_frame_wrapper, task): task for task in tasks}
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    processed_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to process: {task[0]}")
                
                # Print progress every 100 frames
                if (processed_count + failed_count) % 100 == 0:
                    print(f"Progress: {processed_count + failed_count}/{len(frame_files)} frames processed")
                    
            except Exception as exc:
                failed_count += 1
                print(f"Frame {task[0]} generated an exception: {exc}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} frames")
    print(f"Failed: {failed_count} frames")
    print(f"Output folder: {output_folder}")

def process_frame_wrapper(task):
    """Wrapper function for multiprocessing - unpacks the task tuple"""
    input_path, output_path, bw_mode = task
    try:
        process_frame(input_path, output_path, bw_mode)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def write_frame_to_binary(frame_rle, output_file, audio_samples=None):
    """Write a single frame's RLE data to binary file with audio samples"""
    audio_index = 0
    
    for row_index, row_rle in enumerate(frame_rle):
        for color, length in row_rle:
            # Write pixel instruction (3 bytes: length in upper 24 bits, color in lower 8 bits)
            instruction_bytes = utils.pixel_instruction(length, color)
            output_file.write(instruction_bytes)
        
        # Write audio sample after each row (480 rows total)
        if audio_samples and audio_index < len(audio_samples):
            audio_instruction_bytes = utils.audio_instruction(audio_samples[audio_index])
            output_file.write(audio_instruction_bytes)
            audio_index += 1
    
    # Write remaining 45 audio samples at the end of the frame
    # Total: 480 (after rows) + 45 (at end) = 525 audio samples per frame
    remaining_samples = 45
    for _ in range(remaining_samples):
        if audio_samples and audio_index < len(audio_samples):
            audio_instruction_bytes = utils.audio_instruction(audio_samples[audio_index])
            output_file.write(audio_instruction_bytes)
            audio_index += 1
        else:
            # If no audio data available, write silence (0)
            audio_instruction_bytes = utils.audio_instruction(0)
            output_file.write(audio_instruction_bytes)

def convert_frames_to_binary(output_binary_path, include_audio=False, audio_folder=None, use_pwm_ramp=False, pwm_max_value=255):
    """Convert all modified frames to a single binary file with RLE instructions"""
    input_folder = constants.MODIFIED_FRAMES_FOLDER
    
    if not os.path.exists(input_folder):
        print(f"Modified frames folder {input_folder} does not exist. Run frame processing first.")
        return False
    
    # Get list of frame files
    frame_files = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_files.append(filename)
    
    if not frame_files:
        print(f"No modified frames found in {input_folder}")
        return False
    
    print(f"Converting {len(frame_files)} frames to binary format...")
    
    # Load audio data if available
    audio_data = None
    if include_audio:
        if use_pwm_ramp:
            # Generate PWM ramp pattern
            total_samples_needed = len(frame_files) * 525
            audio_data = []
            current_value = 0
            
            for i in range(total_samples_needed):
                audio_data.append(current_value)
                current_value = (current_value + 1) % (pwm_max_value + 1)
            
            print(f"Generated PWM ramp audio data: {len(audio_data)} samples (0 to {pwm_max_value})")
        
        elif audio_folder and os.path.exists(audio_folder):
            audio_file_path = os.path.join(audio_folder, "audio.wav")
            if os.path.exists(audio_file_path):
                try:
                    import wave
                    with wave.open(audio_file_path, 'rb') as wav_file:
                        audio_frames = wav_file.readframes(-1)
                        audio_data = list(audio_frames)
                    print(f"Loaded audio data: {len(audio_data)} samples")
                except Exception as e:
                    print(f"Failed to load audio: {e}")
                    audio_data = None
    
    with open(output_binary_path, 'wb') as binary_file:
        for i, filename in enumerate(frame_files):
            frame_path = os.path.join(input_folder, filename)
            
            # Load and convert frame to RLE
            img = Image.open(frame_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pixels = np.array(img)
            height, width = pixels.shape[:2]
            
            # Convert to 8-bit format
            pixels_8bit = utils.rgb_to_8bit(pixels, bw_mode=False)
            
            # Generate RLE for this frame
            frame_rle = []
            for y in range(height):
                row_rle = []
                run_length = 1
                previous_colour = pixels_8bit[y, 0]
                
                for x in range(1, width):
                    current_colour = pixels_8bit[y, x]
                    if current_colour == previous_colour:
                        run_length += 1
                    else:
                        row_rle.append((previous_colour, run_length))
                        previous_colour = current_colour
                        run_length = 1
                row_rle.append((previous_colour, run_length))
                frame_rle.append(row_rle)
            
            # Prepare 525 audio samples for this frame
            frame_audio_samples = None
            if audio_data:
                start_sample = i * 525
                end_sample = start_sample + 525
                if start_sample < len(audio_data):
                    frame_audio_samples = audio_data[start_sample:end_sample]
                    # Pad with silence if needed
                    while len(frame_audio_samples) < 525:
                        frame_audio_samples.append(0)
            
            # Write frame RLE data to binary file with audio
            write_frame_to_binary(frame_rle, binary_file, frame_audio_samples)
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(frame_files)} frames")
        
        # Write stop instruction at the end
        stop_bytes = utils.stop_instruction()
        binary_file.write(stop_bytes)
    
    print(f"Binary conversion complete! Output saved to: {output_binary_path}")
    print(f"Binary file size: {os.path.getsize(output_binary_path)} bytes")
    if include_audio and use_pwm_ramp:
        print(f"PWM ramp audio included: 0 to {pwm_max_value}, 525 samples per frame")
    return True

def convert_frames_to_binary_concurrent(output_binary_path, include_audio=False, audio_folder=None, max_workers=None, use_pwm_ramp=False, pwm_max_value=255):
    """Convert frames to binary using concurrent processing for better performance"""
    input_folder = constants.MODIFIED_FRAMES_FOLDER
    
    if not os.path.exists(input_folder):
        print(f"Modified frames folder {input_folder} does not exist. Run frame processing first.")
        return False
    
    # Get list of frame files
    frame_files = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_files.append(filename)
    
    if not frame_files:
        print(f"No modified frames found in {input_folder}")
        return False
    
    print(f"Converting {len(frame_files)} frames to binary format using concurrent processing...")
    
    # Load audio data if available
    audio_data = None
    if include_audio:
        if use_pwm_ramp:
            # Generate PWM ramp pattern
            total_samples_needed = len(frame_files) * 525
            audio_data = []
            current_value = 0
            
            for i in range(total_samples_needed):
                audio_data.append(current_value)
                current_value = (current_value + 1) % (pwm_max_value + 1)
            
            print(f"Generated PWM ramp audio data: {len(audio_data)} samples (0 to {pwm_max_value})")
        
        elif audio_folder and os.path.exists(audio_folder):
            audio_file_path = os.path.join(audio_folder, "audio.wav")
            if os.path.exists(audio_file_path):
                try:
                    import wave
                    with wave.open(audio_file_path, 'rb') as wav_file:
                        audio_frames = wav_file.readframes(-1)
                        audio_data = list(audio_frames)
                    print(f"Loaded audio data: {len(audio_data)} samples")
                except Exception as e:
                    print(f"Failed to load audio: {e}")
                    audio_data = None
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(frame_files))
    
    print(f"Using {max_workers} workers for processing")
    
    # Process frames concurrently to generate RLE data
    tasks = []
    for filename in frame_files:
        frame_path = os.path.join(input_folder, filename)
        tasks.append(frame_path)
    
    frame_rle_data = [None] * len(frame_files)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(process_frame_to_rle, task): i for i, task in enumerate(tasks)}
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                frame_rle = future.result()
                frame_rle_data[index] = frame_rle
                
                if (index + 1) % 100 == 0:
                    print(f"Processed {index + 1}/{len(frame_files)} frames")
                    
            except Exception as exc:
                print(f"Frame {tasks[index]} generated an exception: {exc}")
                return False
    
    # Write all frame data to binary file sequentially (to maintain order)
    print("Writing binary data to file...")
    with open(output_binary_path, 'wb') as binary_file:
        for i, frame_rle in enumerate(frame_rle_data):
            if frame_rle is None:
                print(f"Skipping frame {i} due to processing error")
                continue
            
            # Prepare 525 audio samples for this frame
            frame_audio_samples = None
            if audio_data:
                start_sample = i * 525
                end_sample = start_sample + 525
                if start_sample < len(audio_data):
                    frame_audio_samples = audio_data[start_sample:end_sample]
                    # Pad with silence if needed
                    while len(frame_audio_samples) < 525:
                        frame_audio_samples.append(0)
            
            write_frame_to_binary(frame_rle, binary_file, frame_audio_samples)
        
        # Write stop instruction at the end
        stop_bytes = utils.stop_instruction()
        binary_file.write(stop_bytes)
    
    print(f"Binary conversion complete! Output saved to: {output_binary_path}")
    print(f"Binary file size: {os.path.getsize(output_binary_path)} bytes")
    if include_audio and use_pwm_ramp:
        print(f"PWM ramp audio included: 0 to {pwm_max_value}, 525 samples per frame")
    return True

def process_frame_to_rle(frame_path):
    """Process a single frame to RLE format - for use with multiprocessing"""
    try:
        img = Image.open(frame_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = np.array(img)
        height, width = pixels.shape[:2]
        
        # Convert to 8-bit format
        pixels_8bit = utils.rgb_to_8bit(pixels, bw_mode=False)
        
        # Generate RLE for this frame
        frame_rle = []
        for y in range(height):
            row_rle = []
            run_length = 1
            previous_colour = pixels_8bit[y, 0]
            
            for x in range(1, width):
                current_colour = pixels_8bit[y, x]
                if current_colour == previous_colour:
                    run_length += 1
                else:
                    row_rle.append((previous_colour, run_length))
                    previous_colour = current_colour
                    run_length = 1
            row_rle.append((previous_colour, run_length))
            frame_rle.append(row_rle)
        
        return frame_rle
        
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}")
        return None