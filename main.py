from rle_codec import *
from utils import *
from video import *
import constants

'''
Workflow:
1. Process the video into frames and wave file with ffmpeg
2. Run RLE encoding on the frames and create modified video result
3. If result is satisfactory, save the output as binary file
'''

def main():
    print("Select operation:")
    print("1: Extract frames and audio from video")
    print("2: Process frames into RLE ")
    print("3: Create video preview from modified frames (Optional)")
    print("4: Generate RLE binary file from modified frames")
    operation = input("Enter 1, 2, 3, or 4: ").strip()

    if operation == "1":
        print("Input video path:")
        video_path = input().strip()
        print("Select colour mode:")
        print("1: Normal colour")
        print("2: Black and white")
        quant_mode = input("Enter 1 or 2: ").strip()
        bw_mode = (quant_mode == "2")

        extract_audio(video_path, output_folder=constants.AUDIO_FOLDER)
        extract_frames(video_path, output_folder=constants.FRAMES_FOLDER, bw_mode=bw_mode)

    elif operation == "2":
        print("Select colour mode:")
        print("1: Normal colour")
        print("2: Black and white")
        quant_mode = input("Enter 1 or 2: ").strip()
        bw_mode = (quant_mode == "2")

        # Ask for similarity threshold
        from rle_codec import SIMILARITY_THRESHOLD
        print(f"Current similarity threshold: {SIMILARITY_THRESHOLD}")
        user_threshold = input("Enter similarity threshold (0.0-1.0, press Enter to keep default): ").strip()
        if user_threshold:
            try:
                import rle_codec
                rle_codec.SIMILARITY_THRESHOLD = float(user_threshold)
                print(f"Set similarity threshold to {rle_codec.SIMILARITY_THRESHOLD}")
            except Exception as e:
                print(f"Invalid input, using default. Error: {e}")

        process_all_frames_concurrent(bw_mode=bw_mode)

    elif operation == "3":
        create_video(input_frames_folder=constants.MODIFIED_FRAMES_FOLDER, input_audio_folder=constants.AUDIO_FOLDER, output_path="output_video.mp4")
        
    elif operation == "4":
        print("Generate RLE binary file")
        
        # Check if modified frames exist
        import os
        if not os.path.exists(constants.MODIFIED_FRAMES_FOLDER):
            print(f"Error: Modified frames folder {constants.MODIFIED_FRAMES_FOLDER} does not exist.")
            print("Please run operation 2 (Process frames) first.")
            return
        
        # Ask for output binary file path
        default_output = "output.rle"
        output_path = input(f"Enter output binary file path (default: {default_output}): ").strip()
        if not output_path:
            output_path = default_output
        
        # Ask if user wants to include audio
        include_audio = input("Include audio in binary file? (y/n): ").strip().lower() == 'y'
        audio_folder = None
        use_pwm_ramp = False
        pwm_max_value = 255
        
        if include_audio:
            print("Select audio source:")
            print("1: WAV file from audio folder")
            print("2: PWM ramp pattern (0 to max value)")
            audio_source = input("Enter 1 or 2: ").strip()
            
            if audio_source == "2":
                use_pwm_ramp = True
                pwm_input = input("Enter PWM max value (default: 255): ").strip()
                if pwm_input:
                    try:
                        pwm_max_value = int(pwm_input)
                        if pwm_max_value < 1 or pwm_max_value > 255:
                            print("PWM max value must be between 1 and 255, using default 255")
                            pwm_max_value = 255
                    except ValueError:
                        print("Invalid PWM value, using default 255")
                        pwm_max_value = 255
                print(f"Using PWM ramp: 0 to {pwm_max_value}")
            else:
                audio_folder = constants.AUDIO_FOLDER
                if not os.path.exists(audio_folder):
                    print(f"Warning: Audio folder {audio_folder} does not exist.")
                    include_audio_confirm = input("Continue without audio? (y/n): ").strip().lower() == 'y'
                    if not include_audio_confirm:
                        return
                    include_audio = False
        
        # Ask for processing method
        print("Select processing method:")
        print("1: Sequential (slower but uses less memory)")
        print("2: Concurrent (faster but uses more memory)")
        method = input("Enter 1 or 2: ").strip()
        
        if method == "2":
            # Concurrent processing
            max_workers = input("Enter number of workers (press Enter for auto): ").strip()
            if max_workers:
                try:
                    max_workers = int(max_workers)
                except ValueError:
                    print("Invalid input, using auto worker count")
                    max_workers = None
            else:
                max_workers = None
            
            success = convert_frames_to_binary_concurrent(
                output_path, 
                include_audio=include_audio, 
                audio_folder=audio_folder,
                max_workers=max_workers,
                use_pwm_ramp=use_pwm_ramp,
                pwm_max_value=pwm_max_value
            )
        else:
            # Sequential processing
            success = convert_frames_to_binary(
                output_path, 
                include_audio=include_audio, 
                audio_folder=audio_folder,
                use_pwm_ramp=use_pwm_ramp,
                pwm_max_value=pwm_max_value
            )
        
        if success:
            print(f"Successfully generated RLE binary file: {output_path}")
            if include_audio:
                if use_pwm_ramp:
                    print(f"PWM ramp audio included: 0 to {pwm_max_value}, 525 samples per frame")
                else:
                    print("WAV file audio included: 525 samples per frame (480 after rows + 45 at end)")
        else:
            print("Failed to generate RLE binary file")
    else:
        print("Invalid operation selected.")

if __name__ == "__main__":
    main()