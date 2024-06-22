import tenseal as ts
import base64
import os

'''
This file contains code for performing encryption, decryption and executing eval function
using the ckks scheme module of tenseal library. It contains functions to create key,
encrypt data with private key, decrypt data with private key and perform eval function on
the encrypted data using the public key.

Additionally it also provides functions to store the private and public key
in Secret and Public folder respectively and the encrypted data in Result and registration
folder in base64 format.
'''

def write_data(file_path,file_name, data):
    # print ("Writing: ", file_path,"/",file_name)
    if file_path and not os.path.exists(file_path):
        os.makedirs(file_path)
        
    if type(data) == bytes:
        #bytes to base64
        data = base64.b64encode(data)
        
    file_name = file_path + '/' + file_name     
    with open(file_name, 'wb') as f: 
        f.write(data)
 
def read_data(file_path, file_name):
    # print ("Reading: ", file_path,"/",file_name)
    file_name = file_path + '/' + file_name
    with open(file_name, 'rb') as f:
        data = f.read()
    #base64 to bytes
    return base64.b64decode(data)

def create_key():
    context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree = 8192,
                    coeff_mod_bit_sizes = [60, 40, 40, 60]
                   )

    context.generate_galois_keys()
    context.global_scale = 2**40

    secret_context = context.serialize(save_secret_key = True)
    write_data('Secret', 'secret.txt' , secret_context)

    context.make_context_public() #drop the secret_key from the context
    public_context = context.serialize()
    write_data('Public', 'public.txt' , public_context)

    del context, secret_context, public_context

# EncryptionandStore
'''
This function encrypts the iris code and stores the encrypted data in base64 encoding in 
the Registration folder.
'''
def EncryptandStore(id, vector):
    # Load secret key and context
    context = ts.context_from(read_data("Secret","secret.txt"))

    # Create CKKS vectors from the iris code
    enc_v1 = ts.ckks_vector(context, vector)

    # Serialize and save encrypted vectors
    enc_v1_proto = enc_v1.serialize()

    id+= '.txt'
    write_data("Registration", id, enc_v1_proto)

    del context, enc_v1, enc_v1_proto
    
#eval
'''
This function is to perform the Xoring of the encrytped Registed iris code with the 
Calculated encrypted iris code to be authenticated. 
The result of the xoring is then stored in the result folder.
'''
def eval(id1, id2):
    # print("id1: ",id1)
    # print("id2: ", id2)
    context = ts.context_from(read_data('Public', 'public.txt'))#public context
    id1 += '.txt' 
    enc_v1_proto = read_data("Registration", id1)
    enc_v1 = ts.lazy_ckks_vector_from(enc_v1_proto)
    enc_v1.link_context(context)

    id2 += '.txt'
    enc_v2_proto = read_data("Registration", id2)
    enc_v2 = ts.lazy_ckks_vector_from(enc_v2_proto)
    enc_v2.link_context(context)
    
    # xor for binary  = (a-b)^2 = (a-b)(a-b) = a + b -2ab(a and b are not squared bcoz binary)
    
    # sum_result = enc_v2 + enc_v1
    # mul_result = enc_v1 * enc_v2
    # mul2_result = mul_result * 2
    # xor_result = sum_result - mul2_result
    
    diff_result = enc_v2 - enc_v1
    sq_result = diff_result * diff_result
    xor_result = sq_result

    # print('xorresult: ',xor_result)
    write_data('Result',id1,xor_result.serialize())
    
    # del context,mul_result,  mul2_result,enc_v1, enc_v2, enc_v1_proto, enc_v2_proto, sum_result, xor_result
    del context,sq_result, enc_v1, enc_v2, enc_v1_proto, enc_v2_proto, diff_result, xor_result
    


#decryption
'''
Stored encrypted Xoring result is decrypted and returned.
'''
def decryption(id_txt):
    context = ts.context_from(read_data('Secret', 'secret.txt'))
    result_read = read_data('Result',id_txt)
    final_xor = ts.lazy_ckks_vector_from(result_read)
    final_xor.link_context(context)
    xored = final_xor.decrypt()
    # print('before rounding xored: ', xored)
    binary_result = [round(x) for x in xored]
    # print('xored: ', binary_result)
    return binary_result




# #For testing run this file alone.
# vector1 = [0, 0, 1, 1]
# vector2 = [0, 1, 0, 1]
# print('type: ', type(vector1))

# id1= 'test1'
# id2 = 'test2'
# EncryptandStore(id1, vector1 )
# EncryptandStore(id2, vector2 )

# eval(id1, id2)

# id1_txt = id1 + '.txt'

# print('xored binary decrypted:', decryption(id1_txt))




