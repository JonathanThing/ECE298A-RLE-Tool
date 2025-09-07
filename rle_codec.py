import struct
from PIL import Image
import numpy as np
import os
import concurrent.futures
from multiprocessing import cpu_count

import constants
import utils

# Similarity threshold when merging strips
SIMILARITY_THRESHOLD = 0.7 
BW_MODE = True

def verify_rle(frame_rle):
    for row in frame_rle:
        if not utils.verify_rle(row):
            print("RLE verification failed for row:", row)
            return False
    return True

def set_rle_length(row_rle, index, new_length):
    colour = row_rle[index][0]
    row_rle[index] = (colour, new_length)

def increment_rle_length(row_rle, index, increment):
    colour, current_length = row_rle[index]
    row_rle[index] = (colour, current_length + increment)

def enforce_rle_constraints(row_rle):
    row_rle = remove_short_strips(row_rle)
    row_rle = enforce_group_constraints(row_rle)
    return row_rle

# Remove strips less than 3 pixels
def remove_short_strips(row_rle):
    i = len(row_rle) - 1   # Start from right most strip and work leftwards

    while i >= 0:
        colour, length = row_rle[i]
        if length < 3:
            if i == len(row_rle) - 1:                       # If at the rightmost strip, only merge left
                increment_rle_length(row_rle, i-1, length)
                del row_rle[i]
            elif i == 0:                                    # If at the leftmost strip, only merge right
                increment_rle_length(row_rle, i+1, length)
                del row_rle[i]
            else:                                
                absorb_strip_into_neighbor(row_rle, i)
            i -= 1
            continue

        i -= 1
    return row_rle

# Absorb strip into a neighbor
def absorb_strip_into_neighbor(row_rle, index):
    colour, length = row_rle[index]
    
    merge_decision = None # -1 for left neighbour, 1 for right neighbour

    # Check bounds
    left_available = index > 0
    right_available = index < len(row_rle) - 1
    
    if left_available and right_available:
        left_colour, left_length = row_rle[index-1]
        right_colour, right_length = row_rle[index+1]
        
        # Try to merge with same colour neighbor
        if left_colour == colour:
            merge_decision = -1
        elif right_colour == colour:
            merge_decision = 1
        else:
            if not BW_MODE: 
                # Colour (can merge based on similarity)
                left_similarity = calculate_colour_similarity(colour, left_colour)
                right_similarity = calculate_colour_similarity(colour, right_colour)
                
                # Merge if similar enough
                if left_similarity >= SIMILARITY_THRESHOLD and left_similarity >= right_similarity: 
                    merge_decision = -1
                elif right_similarity >= SIMILARITY_THRESHOLD:
                    merge_decision = 1
                else:
                    # If neither are similar enough, just merge with larger strip to reduce impact
                    merge_target = -1 if left_length >= right_length else +1
            else: 
                # Black and White, Can only merge based on larger neighbour              
                merge_target = -1 if left_length >= right_length else +1

    elif left_available: # No other options
        merge_decision = -1
    elif right_available: # No other options
        merge_decision = 1
    else: # Not possible to merge
        return 

    increment_rle_length(row_rle, index + merge_decision, length)
    
    del row_rle[index]

# Compare how similar two colours are by counting bits
def calculate_colour_similarity(colour1, colour2):
    if colour1 == colour2:
        return 1.0
    
    differing_bits = bin(colour1 ^ colour2).count('1')
    similarity = 1.0 - (differing_bits / 8.0)
    
    return similarity

# Every consequtive 6 pixel strips must be at least 36 pixels long
def enforce_group_constraints(row_rle):
    i = 0
    last_strip = max(0, len(row_rle) - constants.GROUP_SIZE)

    while i <= last_strip:
        window = row_rle[i:i+constants.GROUP_SIZE]
        total = sum([strip[1] for strip in window])

        if total < constants.MIN_GROUP_LENGTH:
            deficit = constants.MIN_GROUP_LENGTH - total

            # Try to overwite and merge strips in the window
            if try_merge_in_window(row_rle, i, deficit):
                last_strip = max(0, len(row_rle) - constants.GROUP_SIZE) # Update bounds
                continue

            # Distribute the deficit in the window
            extend_window_strips(row_rle, i, deficit)
        i += 1
    return row_rle
 
# Try to merge same colour strips together by overwriting strips inbetween 
def try_merge_in_window(row_rle, start_index, deficit):
    end_index = start_index + constants.GROUP_SIZE
    
    for i in range(start_index, min(end_index - 1, len(row_rle) - 1)):
        for j in range(i + 1, min(end_index, len(row_rle))):
            if row_rle[i][0] == row_rle[j][0]:
                # Calculate the number of pixels inbetween that will change
                merge_cost = sum(row_rle[k][1] for k in range(i + 1, j))
                
                # Want to minimize the amount of pixels changed, do not change more than nesscary
                if merge_cost <= deficit: 
                    merge_strips_range(row_rle, i, j)
                    return True
    
    return False

# Merge the two strips together, overwriting the strips inbetween them
def merge_strips_range(row_rle, start, end):
    total_length = sum(row_rle[k][1] for k in range(start, end + 1))
    colour = row_rle[start][0]      

    for i in range(end - start):
        del row_rle[start + 1]

    row_rle[start] = (colour, total_length)

# Extends the strips inside the window until the deficit is met, taking from strips outside of the window 
def extend_window_strips(row_rle, start_index, deficit):
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
        for colour, length in row_rle:
            # Write pixel instruction (3 bytes: length in upper 24 bits, colour in lower 8 bits)
            instruction_bytes = utils.pixel_instruction(length, colour)
            output_file.write(instruction_bytes)
        
        # Write audio sample after each row (480 rows total)
        if audio_samples and audio_index < len(audio_samples):
            audio_instruction_bytes = utils.audio_instruction(audio_samples[audio_index])
            output_file.write(audio_instruction_bytes)
            audio_index += 1
    
    # Write remaining 45 audio samples at the end of the frame
    # Total: 480 (after rows) + 45 (at end) = 525 audio samples per frame
    remaining_samples = 45
    for i in range(remaining_samples):
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