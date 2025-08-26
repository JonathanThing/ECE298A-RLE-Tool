from PIL import Image
import os
import sys

def append_images_vertically(image_paths, output_path="appended_vertical.png"):
    """
    Appends 640x480 images vertically into one long image (640 x n*480).
    """
    images = [Image.open(p) for p in image_paths]
    
    # Check all images are 640x480
    for idx, img in enumerate(images):
        if img.size != (640, 480):
            raise ValueError(f"Image {image_paths[idx]} is not 640x480, but {img.size}")
    
    n = len(images)
    width = 640
    total_height = n * 480
    
    # Create new blank image
    result = Image.new('RGB', (width, total_height))
    
    # Paste each image one below the other
    for i, img in enumerate(images):
        result.paste(img, (0, i * 480))
    
    result.save(output_path)
    print(f"Saved appended image as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python appender_vertical.py <output.png> <img1.png> <img2.png> ...")
        sys.exit(1)
    
    output_file = sys.argv[1]
    image_files = sys.argv[2:]

    append_images_vertically(image_files, output_path=output_file)