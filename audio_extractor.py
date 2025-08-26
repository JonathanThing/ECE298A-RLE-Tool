#!/usr/bin/env python3
"""
Audio Extractor Script

This script extracts the first n*525 samples from a wave file and provides
analysis of the chunks, including separators between every 525 samples
and highlighting the first 480 samples of each chunk.
"""

import wave
import struct
import sys
import os

def extract_audio_samples(input_wave_file, num_frames, output_file=None, show_analysis=True):
    """
    Extract the first n*525 samples from a wave file
    
    Args:
        input_wave_file (str): Path to input WAV file
        num_frames (int): Number of 525-sample frames to extract
        output_file (str): Optional output file to save extracted samples
        show_analysis (bool): Whether to show chunk analysis
    """
    
    if not os.path.exists(input_wave_file):
        print(f"Error: Input file '{input_wave_file}' does not exist")
        return False
    
    try:
        # Open the wave file
        with wave.open(input_wave_file, 'rb') as wav_file:
            # Get audio properties
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            total_frames = wav_file.getnframes()
            
            print(f"Audio Properties:")
            print(f"  Channels: {channels}")
            print(f"  Sample width: {sample_width} bytes")
            print(f"  Frame rate: {framerate} Hz")
            print(f"  Total frames: {total_frames}")
            print(f"  Duration: {total_frames / framerate:.2f} seconds")
            print()
            
            # Calculate total samples needed
            total_samples_needed = num_frames * 525
            
            if total_samples_needed > total_frames:
                print(f"Warning: Requested {total_samples_needed} samples but file only has {total_frames}")
                print(f"Will extract {total_frames} samples instead")
                total_samples_needed = total_frames
                num_frames = total_frames // 525
            
            print(f"Extracting {total_samples_needed} samples ({num_frames} chunks of 525)")
            print()
            
            # Read the required samples
            raw_audio = wav_file.readframes(total_samples_needed)
            
            # Convert to list of sample values
            if sample_width == 1:
                # 8-bit unsigned
                samples = list(struct.unpack(f'{total_samples_needed}B', raw_audio))
            elif sample_width == 2:
                # 16-bit signed
                samples = list(struct.unpack(f'<{total_samples_needed}h', raw_audio))
            else:
                print(f"Unsupported sample width: {sample_width}")
                return False
            
            # Save to output file if specified
            if output_file:
                with open(output_file, 'wb') as out_file:
                    out_file.write(raw_audio)
                print(f"Raw audio data saved to: {output_file}")
                print()
            
            # Show analysis if requested
            if show_analysis:
                show_chunk_analysis(samples, num_frames)
            
            return samples
            
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return False

def show_chunk_analysis(samples, num_frames):
    """Show analysis of 525-sample chunks"""
    
    print("=" * 80)
    print("CHUNK ANALYSIS")
    print("=" * 80)
    
    for frame_idx in range(num_frames):
        start_idx = frame_idx * 525
        end_idx = start_idx + 525
        
        frame_samples = samples[start_idx:end_idx]
        
        print(f"\nFrame {frame_idx + 1}:")
        print(f"  Samples {start_idx} to {end_idx - 1} (525 samples total)")
        
        # Show first 480 samples (after each row)
        row_samples = frame_samples[:480]
        print(f"  Row samples (first 480): {start_idx} to {start_idx + 479}")
        print(f"    Min: {min(row_samples)}, Max: {max(row_samples)}, Avg: {sum(row_samples)/len(row_samples):.1f}")
        
        # Show last 45 samples (end of frame burst)
        burst_samples = frame_samples[480:]
        print(f"  End burst (last 45): {start_idx + 480} to {start_idx + 524}")
        print(f"    Min: {min(burst_samples)}, Max: {max(burst_samples)}, Avg: {sum(burst_samples)/len(burst_samples):.1f}")
        
        # Show separator
        if frame_idx < num_frames - 1:
            print(f"  --- Separator (next frame starts at sample {end_idx}) ---")

def save_samples_with_separators(samples, num_frames, output_file):
    """Save samples to text file with clear separators"""
    
    with open(output_file, 'w') as f:
        f.write("Audio Sample Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for frame_idx in range(num_frames):
            start_idx = frame_idx * 525
            end_idx = start_idx + 525
            
            frame_samples = samples[start_idx:end_idx]
            
            f.write(f"FRAME {frame_idx + 1} (Samples {start_idx}-{end_idx-1})\n")
            f.write("-" * 40 + "\n")
            
            # Row samples (first 480)
            f.write("Row samples (1-480):\n")
            row_samples = frame_samples[:480]
            for i, sample in enumerate(row_samples):
                if i % 20 == 0:
                    f.write(f"\n  Row {i//20 + 1:2d}: ")
                f.write(f"{sample:4d} ")
            f.write("\n\n")
            
            # End burst samples (last 45)
            f.write("End burst samples (481-525):\n  ")
            burst_samples = frame_samples[480:]
            for i, sample in enumerate(burst_samples):
                if i % 15 == 0 and i > 0:
                    f.write(f"\n       ")
                f.write(f"{sample:4d} ")
            f.write("\n\n")
            
            if frame_idx < num_frames - 1:
                f.write("=" * 50 + "\n\n")
    
    print(f"Detailed sample analysis saved to: {output_file}")

def main():
    """Main function with command line interface"""
    
    print("Audio Sample Extractor")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter path to WAV file: ").strip()
    
    if len(sys.argv) > 2:
        try:
            num_frames = int(sys.argv[2])
        except ValueError:
            print("Invalid number of frames")
            return
    else:
        try:
            num_frames = int(input("Enter number of 525-sample frames to extract: ").strip())
        except ValueError:
            print("Invalid input")
            return
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    raw_output = f"{base_name}_extracted_{num_frames}frames.raw"
    analysis_output = f"{base_name}_analysis_{num_frames}frames.txt"
    
    # Extract samples
    samples = extract_audio_samples(input_file, num_frames, raw_output, show_analysis=True)
    
    if samples:
        # Save detailed analysis
        save_samples_with_separators(samples, num_frames, analysis_output)
        
        print(f"\nExtraction complete!")
        print(f"  Raw data: {raw_output}")
        print(f"  Analysis: {analysis_output}")
        print(f"  Total samples extracted: {len(samples)}")
        print(f"  Frame structure: 480 row samples + 45 burst samples = 525 per frame")

if __name__ == "__main__":
    main()
