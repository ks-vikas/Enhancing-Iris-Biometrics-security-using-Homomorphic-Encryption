import encode_iris as img
import encrypt as en
import os
import argparse

'''
This function deletes the temporary file created in the process.
'''
def delete_txt_file(file_name):
    # print ("Deleting: ", "Registration/",file_name)
    file_path = os.path.join("Registration", file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
'''
This function calls the create_key function to create the keys if they
not exists already.
'''
def createKey():
    if os.path.exists('Secret/secret.txt'):
        if os.path.exists('Public/public.txt'):
            # print("Key already exists thus not creating new key.")
            return
    
    # print("Key not exists thus creating new key.")
    en.create_key()

#Registering the image in the database with its ID
def image_registration(id, image_path):
    encoded_image_vector = img.encode(image_path)
    # print('encode image vector inside iris main type: ', type(encoded_image_vector))
    en.EncryptandStore(id, encoded_image_vector)
 
#Calculate the hamming distance in the binary string    
def hamming_distance(binary):
    hd = 0
    for bit in binary:
        if bit == 1:
            hd +=1
    # print("xored_hd: ", hd)
    return hd     

'''
This function is to authenticate the image present @ the image_path with the 
Registered image in the database with ID == id1
'''   
def authentication(id1, image_path):
    id2 = id1 + '2' # creating temporary name for image to be authenticated.
    
    encoded_image_vector = img.encode(image_path) #iris code for the image to be authenticated
    # print('encode image vector inside iris main type: ', type(encoded_image_vector[0]))
    en.EncryptandStore(id2, encoded_image_vector)
    
    en.eval(id1, id2)  #perform xor for authentication
    result_txt = id1 + '.txt'
    xored_binary = en.decryption(result_txt)
    # print("Xored after decrytpion: ", xored_binary)
    
    delete_txt_file(id2 + '.txt') #remove the temporary file
    hd = hamming_distance(xored_binary)
    hd_res = hd / len(xored_binary)
    # print("hamming distance(hd): ", hd, " length of hd vector: ", len(xored_binary)) 
    # print("hd_result: ", hd_res)
    print(hd_res)
  
'''
Uncomment the following code when not using for testing.
''' 
    # print(hd_res, end = ', ')   
    # matching_threshold = 0.409 #set it equal to the CERR value 
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

'''
Comment out the following code when not using for testing.
'''
def reg_auth(option, id, image_path):


    
    createKey()
    
    if option == 1:
        image_registration(id, image_path)
    elif option == 2:
        authentication(id, image_path)   
        
    return         
                