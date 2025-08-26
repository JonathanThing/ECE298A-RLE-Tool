"""
Audio utilities for extracting and analyzing wave file samples
in the context of the BadApple project's 525-sample frame structure.
"""

import wave
import struct
import os

def extract_frame_samples(wave_file_path, num_frames):
    """
    Extract samples from a wave file in 525-sample chunks
    
    Args:
        wave_file_path (str): Path to the WAV file
        num_frames (int): Number of 525-sample frames to extract
        
    Returns:
        tuple: (samples_list, audio_info_dict) or (None, None) on error
    """
    try:
        with wave.open(wave_file_path, 'rb') as wav_file:
            # Get audio properties
            audio_info = {
                'channels': wav_file.getnchannels(),
                'sample_width': wav_file.getsampwidth(),
                'framerate': wav_file.getframerate(),
                'total_frames': wav_file.getnframes()
            }
            
            # Calculate samples needed
            total_samples_needed = num_frames * 525
            
            if total_samples_needed > audio_info['total_frames']:
                total_samples_needed = audio_info['total_frames']
                num_frames = total_samples_needed // 525
            
            # Read raw audio data
            raw_audio = wav_file.readframes(total_samples_needed)
            
            # Convert to sample values based on bit depth
            if audio_info['sample_width'] == 1:
                # 8-bit unsigned (0-255)
                samples = list(struct.unpack(f'{total_samples_needed}B', raw_audio))
            elif audio_info['sample_width'] == 2:
                # 16-bit signed (-32768 to 32767)
                samples = list(struct.unpack(f'<{total_samples_needed}h', raw_audio))
                # Convert to 8-bit range (0-255) for compatibility
                samples = [(s + 32768) // 256 for s in samples]
            else:
                raise ValueError(f"Unsupported sample width: {audio_info['sample_width']}")
            
            audio_info['extracted_samples'] = len(samples)
            audio_info['extracted_frames'] = len(samples) // 525
            
            return samples, audio_info
            
    except Exception as e:
        print(f"Error extracting audio samples: {e}")
        return None, None

def split_frame_samples(samples, frame_index):
    """
    Split a 525-sample frame into row samples (480) and burst samples (45)
    
    Args:
        samples (list): List of all samples
        frame_index (int): Frame index (0-based)
        
    Returns:
        tuple: (row_samples, burst_samples)
    """
    start_idx = frame_index * 525
    end_idx = start_idx + 525
    
    if end_idx > len(samples):
        raise IndexError(f"Frame {frame_index} extends beyond available samples")
    
    frame_samples = samples[start_idx:end_idx]
    row_samples = frame_samples[:480]  # First 480 samples (after each row)
    burst_samples = frame_samples[480:]  # Last 45 samples (end of frame)
    
    return row_samples, burst_samples

def get_frame_samples(samples, frame_index):
    """
    Get all 525 samples for a specific frame
    
    Args:
        samples (list): List of all samples
        frame_index (int): Frame index (0-based)
        
    Returns:
        list: 525 samples for the frame
    """
    start_idx = frame_index * 525
    end_idx = start_idx + 525
    
    if end_idx > len(samples):
        raise IndexError(f"Frame {frame_index} extends beyond available samples")
    
    return samples[start_idx:end_idx]

def analyze_frame_audio(samples, frame_index):
    """
    Analyze audio characteristics for a specific frame
    
    Args:
        samples (list): List of all samples
        frame_index (int): Frame index (0-based)
        
    Returns:
        dict: Analysis results
    """
    row_samples, burst_samples = split_frame_samples(samples, frame_index)
    
    analysis = {
        'frame_index': frame_index,
        'total_samples': 525,
        'row_samples': {
            'count': len(row_samples),
            'min': min(row_samples),
            'max': max(row_samples),
            'avg': sum(row_samples) / len(row_samples),
            'samples': row_samples
        },
        'burst_samples': {
            'count': len(burst_samples),
            'min': min(burst_samples),
            'max': max(burst_samples),
            'avg': sum(burst_samples) / len(burst_samples),
            'samples': burst_samples
        }
    }
    
    return analysis

def save_extracted_audio(samples, output_path, sample_width=1):
    """
    Save extracted samples as a new WAV file
    
    Args:
        samples (list): List of sample values
        output_path (str): Output WAV file path
        sample_width (int): Bytes per sample (1 or 2)
    """
    try:
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(44100)  # Standard sample rate
            
            if sample_width == 1:
                # 8-bit unsigned
                audio_data = struct.pack(f'{len(samples)}B', *samples)
            else:
                # 16-bit signed - convert from 8-bit range
                samples_16bit = [(s * 256) - 32768 for s in samples]
                audio_data = struct.pack(f'<{len(samples_16bit)}h', *samples_16bit)
            
            wav_file.writeframes(audio_data)
            
        print(f"Extracted audio saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

# Example usage functions
def extract_for_video_frames(wave_file_path, num_video_frames, output_dir="output"):
    """
    Extract audio samples for a specific number of video frames
    This is the main function you'd use in your video processing pipeline
    """
    print(f"Extracting audio for {num_video_frames} video frames...")
    
    samples, audio_info = extract_frame_samples(wave_file_path, num_video_frames)
    
    if samples is None:
        return None
    
    print(f"Audio Info:")
    print(f"  Sample rate: {audio_info['framerate']} Hz")
    print(f"  Bit depth: {audio_info['sample_width'] * 8} bits")
    print(f"  Total samples extracted: {len(samples)}")
    print(f"  Frame structure: 480 row + 45 burst = 525 samples per frame")
    
    # Save raw samples for use in RLE encoder
    os.makedirs(output_dir, exist_ok=True)
    raw_output = os.path.join(output_dir, "audio_samples.raw")
    
    with open(raw_output, 'wb') as f:
        # Save as bytes (0-255 range)
        f.write(bytes(samples))
    
    print(f"Raw audio data saved to: {raw_output}")
    
    return samples
