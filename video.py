import constants
import subprocess
import os
import sys
from PIL import Image
import wave
import numpy as np

def extract_frames(video_path, output_folder=constants.FRAMES_FOLDER, bw_mode=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.listdir(output_folder):
        response = input("Frames already exist. Do you want to delete them? (y/n): ").strip().lower()
        if response == "y" or response == "yes":
            for filename in os.listdir(output_folder):
                if filename.endswith(".png"):
                    os.remove(os.path.join(output_folder, filename))
        else:
            print("Cancelled frame extraction.")
            return

    if bw_mode:
        vf_args = "format=gray,lut=y='if(gte(val,128),255,0)',fps=60,scale=640:480"
    else:
        vf_args = "lutrgb=r='trunc(val/32)*32':g='trunc(val/32)*32':b='trunc(val/64)*64',fps=60,scale=640:480"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf_args,
        os.path.join(output_folder, constants.FRAME_FILE)
    ]
    subprocess.run(cmd, check=True)

def extract_audio(video_path, output_folder=constants.AUDIO_FOLDER):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if constants.AUDIO_FILE in os.listdir(output_folder):
        response = input("Audio file already exists. Do you want to delete it? (y/n): ").strip().lower()
        if response == "y" or response == "yes":
            os.remove(os.path.join(output_folder, constants.AUDIO_FILE))
        else:
            print("Cancelled audio extraction.")
            return

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_u8",
        "-ar", "31468", # Sample Rate: 25.175MHz / 800 = 31.468kHz
        "-ac", "1",
        os.path.join(constants.AUDIO_FOLDER, constants.AUDIO_FILE)
    ]
    subprocess.run(cmd, check=True)

def create_video(input_frames_folder=constants.MODIFIED_FRAMES_FOLDER, input_audio_folder=constants.AUDIO_FOLDER, output_path="output_video.mp4"):
    if not os.path.exists(input_frames_folder):
        print("Modified frames folder does not exist. Cannot create video.")
        return

    cmd = [
        "ffmpeg",
        "-framerate", "60",
        "-i", os.path.join(input_frames_folder, constants.FRAME_FILE),
        "-i", os.path.join(input_audio_folder, constants.AUDIO_FILE),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True)

