import os

def average_file_size(folder_path, file_extension=None):
    total_size = 0
    file_count = 0
    
    for filename in os.listdir(folder_path):
        if file_extension is None or filename.endswith(file_extension):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
                file_count += 1
                
    if file_count == 0:
        return 0
    return total_size / file_count / 1024  # Convert bytes to KB


import cv2

def average_image_size(folder_path):
    total_size = 0
    file_count = 0
    image_sizes = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Common image formats
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Get image size using cv2
                img = cv2.imread(file_path)
                if img is not None:
                    height, width, _ = img.shape
                    image_sizes.append((width, height))  # Append (width, height)
                
                total_size += os.path.getsize(file_path)
                file_count += 1

    if file_count == 0:
        return 0, []

    average_size_kb = total_size / file_count / 1024  # Convert bytes to KB
    return average_size_kb, image_sizes



image_size_kB, image_size = average_image_size("../datasets/dummy/valid")
print(f"Average size of images: {image_size_kB:.2f} KB, the size of the images: {image_size[0][0]}x{image_size[0][1]}")

image_size_kB, image_size = average_image_size("./output/reconstructed")
print(f"Average size of reconstructed images: {image_size_kB:.2f} KB, the size of the images: {image_size[0][0]}x{image_size[0][1]}")


average_bin_size = average_file_size("./output/binary/bin", file_extension=".bin")
print(f"Average size of .bin files in 'bins': {average_bin_size:.2f} KB")