import os
import re
from PIL import Image
from iris_main import reg_auth

def list_image_files(folder_path):
    """List all image files in the given folder path."""
    image_files = []
    for file_name in os.listdir(folder_path):
        if re.match(r'\d{3}[LR]_\d\.png', file_name):  # Regex to match filenames like "001L_1.png"
            image_files.append(file_name)
    image_files.sort()        
    return image_files

def extract_id_from_filename(file_name):
    """Extract the first three digits from the filename as an ID."""
    match = re.match(r'(\d{3})([LR])_\d\.png', file_name)
    if match:
        return match.group(1) + match.group(2)
    return None

def batch_image_registration():
    image_directory_path = "iris_reg" 
    image_files = list_image_files(image_directory_path)
    for file_name in image_files:
        image_id = extract_id_from_filename(file_name)
        if image_id is not None:
            image_path = os.path.join(image_directory_path, file_name)
            # print(f"Image ID: {image_id}")
            reg_auth(1, image_id, image_path)
            # print("Image : ", image_id, " is Registered.")   

def batch_image_authentication():
    image_directory_path = "iris_auth" 
    reg_images_path = "iris_reg"
    image_files = list_image_files(image_directory_path)
    reg_image_files = list_image_files(reg_images_path)
    
    for file_name in image_files:
        # print("#######################################################")
        image_id = extract_id_from_filename(file_name)
        
        if image_id is not None:
            image_path = os.path.join(image_directory_path, file_name)
            
            for reg_file_name in reg_image_files:
                reg_image_id = extract_id_from_filename(reg_file_name)
                # print(f"Authenticating image: {image_id}, with registered image: {reg_image_id}", end = ' ')
                print(f"{image_id},{reg_image_id}", end = ',')
                reg_auth(2, reg_image_id, image_path)    
                
def main():
    batch_image_registration()
    batch_image_authentication()
    


if __name__ == "__main__":
     
    main()               
       
#### usage 
# python3 iris_main.py 1 --id <id> --image_path <image_path>
# python3 iris_main.py 2 --id <id> --image_path <image_path>       