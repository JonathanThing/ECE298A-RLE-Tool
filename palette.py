from PIL import Image
import numpy as np
import os
import sys

def create_8bit_palette():
    """
    Creates a palette of all possible 8-bit colors (3R-3G-2B format)
    Returns a set of RGB tuples for fast lookup
    """
    colors = set()
    
    for r in range(8):  # 3 bits for red (0-7)
        for g in range(8):  # 3 bits for green (0-7)
            for b in range(4):  # 2 bits for blue (0-3)
                # Scale to full 8-bit range (0-255)
                red = (r * 255) // 7
                green = (g * 255) // 7
                blue = (b * 255) // 3
                
                colors.add((red, green, blue))
    
    return colors

def rgb_to_8bit(r, g, b):
    """
    Convert RGB values to nearest 8-bit palette color
    """
    # Convert to 3-3-2 bit values
    r_3bit = round((r * 7) / 255)
    g_3bit = round((g * 7) / 255)
    b_2bit = round((b * 3) / 255)
    
    # Scale back to 8-bit
    r_8bit = (r_3bit * 255) // 7
    g_8bit = (g_3bit * 255) // 7
    b_8bit = (b_2bit * 255) // 3
    
    return (r_8bit, g_8bit, b_8bit)

def check_image_compatibility(image_path, verbose=False):
    """
    Check if an image can be represented using the 8-bit color palette
    Returns (is_compatible, unique_colors, non_palette_colors, total_pixels)
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        
        # Get palette colors
        palette = create_8bit_palette()
        
        # Get unique colors in the image
        unique_colors = set()
        non_palette_colors = set()
        
        height, width = pixels.shape[:2]
        total_pixels = height * width
        
        # Check each unique color
        for y in range(height):
            for x in range(width):
                color = tuple(pixels[y, x])
                unique_colors.add(color)
                
                if color not in palette:
                    non_palette_colors.add(color)
        
        is_compatible = len(non_palette_colors) == 0
        
        if verbose:
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Dimensions: {width}x{height}")
            print(f"Total pixels: {total_pixels}")
            print(f"Unique colors: {len(unique_colors)}")
            print(f"Compatible: {'Yes' if is_compatible else 'No'}")
            
            if not is_compatible:
                print(f"Non-palette colors: {len(non_palette_colors)}")
                if len(non_palette_colors) <= 10:
                    print("Non-palette colors found:")
                    for color in sorted(non_palette_colors):
                        nearest = rgb_to_8bit(*color)
                        print(f"  RGB{color} -> nearest palette: RGB{nearest}")
                else:
                    print(f"Too many non-palette colors to display ({len(non_palette_colors)})")
        
        return is_compatible, len(unique_colors), len(non_palette_colors), total_pixels
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, 0, 0, 0

def convert_to_palette(image_path, output_path=None):
    """
    Convert an image to use only 8-bit palette colors
    """
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        
        height, width = pixels.shape[:2]
        
        # Convert each pixel to nearest palette color
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[y, x]
                pixels[y, x] = rgb_to_8bit(r, g, b)
        
        # Create new image
        converted_img = Image.fromarray(pixels, 'RGB')
        
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_8bit{ext}"
        
        converted_img.save(output_path)
        print(f"Converted image saved as: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return None

def check_folder(folder_path):
    """
    Check all images in a folder for palette compatibility
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    image_files = []
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, filename))
    
    if not image_files:
        print("No image files found in the folder.")
        return
    
    print(f"Checking {len(image_files)} images for 8-bit palette compatibility...\n")
    
    compatible_count = 0
    total_unique_colors = 0
    total_non_palette_colors = 0
    
    for image_path in sorted(image_files):
        is_compatible, unique_colors, non_palette_colors, total_pixels = check_image_compatibility(image_path, verbose=True)
        
        if is_compatible:
            compatible_count += 1
        
        total_unique_colors += unique_colors
        total_non_palette_colors += non_palette_colors
        print("-" * 50)
    
    print(f"\nSummary:")
    print(f"Total images: {len(image_files)}")
    print(f"Compatible images: {compatible_count}")
    print(f"Incompatible images: {len(image_files) - compatible_count}")
    print(f"Compatibility rate: {(compatible_count/len(image_files))*100:.1f}%")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python check_palette_compatibility.py <image_path>")
        print("  python check_palette_compatibility.py <folder_path>")
        print("  python check_palette_compatibility.py <image_path> --convert")
        sys.exit(1)
    
    path = sys.argv[1]
    convert_mode = "--convert" in sys.argv
    
    if os.path.isfile(path):
        # Single image
        is_compatible, unique_colors, non_palette_colors, total_pixels = check_image_compatibility(path, verbose=True)
        
        if convert_mode:
            print("\nConverting to 8-bit palette...")
            convert_to_palette(path)
        elif not is_compatible:
            response = input("\nImage is not compatible. Convert to 8-bit palette? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                convert_to_palette(path)
    
    elif os.path.isdir(path):
        # Folder of images
        check_folder(path)
    
    else:
        print(f"Path not found: {path}")

if __name__ == "__main__":
    main()