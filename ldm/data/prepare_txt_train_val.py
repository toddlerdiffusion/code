import os
import random
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image


def get_files_from_dir(directory_path, saving_text_name, split_percentage = 0.9):
    """
    Generate train and val splits text files.
    Needed by the dataloader.
    """
    # List all files in the directory
    file_names = os.listdir(directory_path)

    # Calculate the number of files for each set
    total_files = len(file_names)
    split_index = int(total_files * split_percentage)
    # Shuffle the list of file names randomly
    random.shuffle(file_names)
    # Split the file names into two sets
    train_set = file_names[:split_index]
    val_set = file_names[split_index:]

    # Create or open a text file for writing
    with open(saving_text_name+"_train.txt", 'w') as file:
        # Write each file name to the text file
        for file_name in train_set:
            file.write(file_name + '\n')

    with open(saving_text_name+"_val.txt", 'w') as file:
        # Write each file name to the text file
        for file_name in val_set:
            file.write(file_name + '\n')

    print('File names have been saved to ', saving_text_name)


def prepare_IN_val(directory_path, img_size, save_dir):
    """
    Read the IN dataset valset and resize it to 256 and save them all in one folder
    """
    folders_names = os.listdir(directory_path)
    for folder in tqdm(folders_names):
        img_names = os.listdir(os.path.join(directory_path, folder))
        for img_name in img_names:
            img_path = os.path.join(directory_path, folder, img_name)
            img = cv2.imread(img_path)
            saving_img_name = img_name.split(".")[0]
            cv2.imwrite(os.path.join(save_dir, saving_img_name+".png"), cv2.resize(img, img_size))


def prepare_COCO_val(directory_path, img_size, save_dir):
    """
    Read the COCO dataset valset and resize it to 256 and save them all in one folder
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_names = os.listdir(directory_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(directory_path, img_name)
        img = cv2.imread(img_path)
        saving_img_name = img_name.split(".")[0]
        cv2.imwrite(os.path.join(save_dir, saving_img_name+".png"), cv2.resize(img, img_size))


def prepare_sketches_val(directory_path, img_size, save_dir, Val_imgnames_txt, binary=False):
    """
    Read the IN dataset valset and resize it to 256 and save them all in one folder
    """
    with open(Val_imgnames_txt) as file:
        img_names = [line.rstrip() for line in file]

    for img_name in tqdm(img_names):
        img_path = os.path.join(directory_path, img_name)
        img = cv2.imread(img_path)
        saving_img_name = img_name.split(".")[0]
        img = cv2.resize(img, img_size)
        if binary:
            # Convert hte image to binary one:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mid_value = (np.max(img) + np.min(img)) / 2
            _, img = cv2.threshold(img, mid_value, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(save_dir, saving_img_name+".png"), img)


def convert_imgs2binary_inplace(directory_path, img_size=None):
    img_names = os.listdir(directory_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(directory_path, img_name)
        img = cv2.imread(img_path)
        if img_size is not None:
            img = cv2.resize(img, img_size)
        # Convert the image to binary one:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mid_value = (float(img.max()) + float(img.min())) / 2
        _, img = cv2.threshold(img, mid_value, 255, cv2.THRESH_BINARY)
        cv2.imwrite(img_path, img)


def convert_imgs2binary(directory_path, saving_dir, img_size=None):
    os.makedirs(saving_dir, exist_ok=True)
    img_names = os.listdir(directory_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(directory_path, img_name)
        img = cv2.imread(img_path)
        if img_size is not None:
            img = cv2.resize(img, img_size)
        # Convert the image to binary one:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mid_value = (float(img.max()) + float(img.min())) / 2
        _, img = cv2.threshold(img, mid_value, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(saving_dir, img_name.replace(".webp", ".jpg")), img)


def save_imgnames_in_dir_txt(directory_path, saving_text_name):
    img_names = os.listdir(directory_path)
    with open(saving_text_name+"_val.txt", 'w') as file:
        # Write each file name to the text file
        for img_name in img_names:
            file.write(img_name + '\n')


def rename_images(path):
    # Ensure the directory exists
    if not os.path.exists(path):
        print(f"Error: Directory '{path}' does not exist.")
        return

    # Iterate over files in the directory
    for filename in os.listdir(path):
        old_path = os.path.join(path, filename)

        # Check if the file is a JPEG image
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
            # Replace the first '0' with '1' in the filename
            new_filename = filename.replace('0', '1', 1)
            
            new_path = os.path.join(path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")


def change_image_extension(directory, target_extension):
    """
    change_image_extension("/path/to/your/images/folder", ".png")
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Iterate over files in the directory
    for filename in tqdm(os.listdir(directory)):
        old_path = os.path.join(directory, filename)

        # Change the file extension
        new_filename = os.path.splitext(filename)[0] + target_extension.lower()
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)


def resize_images(input_dir, output_dir, size):
    """
    Resize images in the input_dir to the specified size and save them to the output_dir.
    
    Parameters:
    - input_dir: Directory containing the original images.
    - output_dir: Directory where resized images will be saved.
    - size: A tuple (width, height) specifying the new size of the images.
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct the full file path
            file_path = os.path.join(input_dir, filename)
            
            # Open and resize the image
            image = Image.open(file_path)
            image_resized = image.resize(size, Image.ANTIALIAS)
            
            # Construct the output file path
            output_file_path = os.path.join(output_dir, filename)
            
            # Save the resized image
            image_resized.save(output_file_path)


def rename_images_in_dir_inplace_add_sample(directory_path):
    """
    Renames all images in the specified directory by adding 'sample_' before each image name.

    Parameters:
    - directory_path: The path to the directory containing the images to be renamed.
    """
    try:
        for filename in os.listdir(directory_path):
            # Check if the file is an image based on its extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Construct the new file name
                new_filename = f"sample_{filename}"
                # Construct the full old and new file paths
                old_file_path = os.path.join(directory_path, filename)
                new_file_path = os.path.join(directory_path, new_filename)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")


def edge_smoothing(input_folder, output_folder):
    # Ensure output folder exists, create it if necessary
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Interpolation method
    interpolation_method = cv2.INTER_LINEAR

    # Loop over each image in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            # Load the image
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Convert the image to binary one:
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            # Resize the image with interpolation
            #smooth_image = cv2.resize(image, (256, 256), interpolation=interpolation_method)
            smooth_image = image

            # Save the smoothed image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, smooth_image)

    print("All images processed and saved.")


if __name__ == '__main__':
    edge_smoothing(input_folder="/home/xxx/CoT_Gen/latent-diffusion-main/data/lsun/churches",
                   output_folder="/home/xxx/CoT_Gen/latent-diffusion-main/data/lsun/churches_Binarysmooth")