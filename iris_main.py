import zzz as img
import encrypt as en
import os
import argparse

def delete_txt_file(file_name):
    # print ("Deleting: ", "Registration/",file_name)
    file_path = os.path.join("Registration", file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

def createKey():
    if os.path.exists('Secret/secret.txt'):
        if os.path.exists('Public/public.txt'):
            # print("Key already exists thus not creating new key.")
            return
    
    # print("Key not exists thus creating new key.")
    en.create_key()

def image_registration(id, image_path):
    encoded_image_vector = img.encode(image_path)
    # print('encode image vector inside iris main type: ', type(encoded_image_vector))
    en.EncryptandStore(id, encoded_image_vector)
    
def hamming_distance(binary):
    hd = 0
    for bit in binary:
        if bit == 1:
            hd +=1
    # print("xored_hd: ", hd)
    return hd        
def authentication(id1, image_path):
    id2 = id1 + '2'
    encoded_image_vector = img.encode(image_path)
    # print('encode image vector inside iris main type: ', type(encoded_image_vector[0]))
    en.EncryptandStore(id2, encoded_image_vector)
    en.eval(id1, id2) 
    result_txt = id1 + '.txt'
    xored_binary = en.decryption(result_txt)
    # print("Xored after decrytpion: ", xored_binary)
    delete_txt_file(id2 + '.txt')
    hd = hamming_distance(xored_binary)
    hd_res = hd / len(xored_binary)
    # print("hamming distance(hd): ", hd, " length of hd vector: ", len(xored_binary)) 
    # print("hd_result: ", hd_res)
    print(hd_res)
    matching_threshold = 0.5
    # if hd_res <= matching_threshold:
    #     print("Authentication Successful")
    # else :
    #     print("Unsuccessful Authentication!!!")  


    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run functions for image registration and authentication.')
#     parser.add_argument('function', choices=['image_registration', 'authentication'], help='Choose the function to run')
#     parser.add_argument('--id', help='ID for image registration or authentication')
#     parser.add_argument('--image_path', help='Path to the image for image registration or authentication')

#     args = parser.parse_args()

    
#     createKey()
    
#     if args.function == 'image_registration':
#         if not args.id or not args.image_path:
#             parser.error("For 'image_registration' function, both --id and --image_path are required.")
#         image_registration(args.id, args.image_path)
#     elif args.function == 'authentication':
#         if not args.id or not args.image_path:
#             parser.error("For 'authentication' function, both --id and --image_path are required.")
#         authentication(args.id, args.image_path)        
        
#### usage 
# python3 iris_main.py image_registration --id <id> --image_path <image_path>
# python3 iris_main.py authentication --id <id> --image_path <image_path>

def reg_auth(option, id, image_path):


    
    createKey()
    
    if option == 1:
        image_registration(id, image_path)
    elif option == 2:
        authentication(id, image_path)   
        
    return         
                