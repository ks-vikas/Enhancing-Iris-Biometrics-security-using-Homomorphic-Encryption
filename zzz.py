import cv2 
import numpy as np
from scipy.signal import convolve2d
from skimage import exposure
import pywt
import utils_preprocess as uh

def encode(im_path):

    image = im_path
    color_image = cv2.imread(image) # read color image
    img = cv2.imread(image,0)   # read gray image
        
    # Find contours to find the frame of the image. inner boundary of the black frame in the images.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # center of the frame and a possible circle for the frame.
    image_center_x = 0
    image_center_y = 0
    boundary_points = None
    iris_boundary_points = []
    
    ################################################## Original image black frame boundary  #################################
    # Ensure at least one contour is found. largest contour will be the frame inner boundary
    if len(contours) > 0:
        # Get the contour with the largest area (outer contour)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour (boundary) on a blank image
        boundary_image = np.zeros_like(img)
        cv2.drawContours(boundary_image, [largest_contour], -1, 255, thickness=1)  # Draw the largest contour
        
        # Display the boundary image (white boundary on black background)
        # cv2.imshow('Boundary Image', boundary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Extract boundary points from the largest contour
        boundary_points = largest_contour.squeeze(axis=1)  # Remove redundant axis
        # print("Boundary Points:")
        # print(boundary_points) 

        points = boundary_points
            # Fit a circle to the points
        image_center_x, image_center_y, radius = uh.fit_circle_to_points(points)

        image= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # cv2.circle(image, (image_center_x,image_center_y), radius, (0,255,0), thickness = 2)
        # cv2.imshow('circle_drawn', image)
        # cv2.waitKey(0)
    
    else:
        print("No iris contour found in the mask image.")

    image = np.copy(img)
    equalized_img = cv2.equalizeHist(image)
    # cv2.imshow('equal Hist', equalized_img) 
    # cv2.waitKey(0)   

    height,width = equalized_img.shape
    img = cv2.medianBlur(equalized_img,5)  # Apply Blur
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) # Create a BGR copy to image for the drawn circles to be visible.
    # cv2.imshow('Equalized Cimg', cimg) 
    # cv2.waitKey(0) 


    ret, thresh1 = cv2.threshold(equalized_img, 200, 255, cv2.THRESH_BINARY_INV) # Threshold the image for iris boundary
    # cv2.imshow('Binary Threshold', thresh1) 
    # cv2.waitKey(0)    

    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.erode(thresh1, kernel, iterations=5)
    # cv2.imshow('Dilated', img_dilation)         

    # # Gaussian Blur 
    Gaussian = cv2.GaussianBlur(img_dilation, (5,5), 0) 
    # cv2.imshow('Gaussian Blurring', Gaussian) 
    # cv2.waitKey(0) 

    iris_boundary_points = uh.traverse_towards_center(Gaussian, boundary_points, image_center_x, image_center_y, error = 25) #find iris_boundary points
    iris_boundary_array = np.array(iris_boundary_points) #Convert to numpy array
    
    # print ('iris array',iris_boundary_array)
    #find iris center and draw iris circle
    
    # Find the center of the iris.
    iris_center_x, iris_center_y, iris_radius = uh.fit_circle_to_points(iris_boundary_array)
    iris_radius  = int(iris_radius * 0.9) 

    # print ('iris circle:',iris_center_x,', ',iris_center_y,', ',iris_radius)
    # newImage= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # cv2.circle(color_image, (iris_center_x, iris_center_y), iris_radius, (0,255,0), thickness = 2)

    # cv2.imshow('iris_circle', color_image)
    # cv2.waitKey(0)

    mask = np.zeros(color_image.shape, dtype=np.uint8)
    cv2.circle(mask, (iris_center_x, iris_center_y), iris_radius, (255, 255, 255), thickness = -1) # Create a mask to hide everything else except iris
    iris_image = cv2.bitwise_and(color_image, mask)
    # cv2.imshow('iris and pupil', iris_image)
    # cv2.circle(iris_image, (iris_center_x, iris_center_y), 3, (255, 0, 0), thickness = -1)
##################################################################################################################################     
    
    # inverted = 255 - iris_image
    # cv2.imshow('inverted', inverted)
    # cv2.waitKey(0)
    
    b,g,r = cv2.split(iris_image)
    gray_img = cv2.cvtColor(iris_image, cv2.COLOR_BGR2GRAY)
    # equalized_img = cv2.equalizeHist(gray_img)
    # ret, threshr = cv2.threshold(r, 40, 255, cv2.THRESH_BINARY)
    equalized_img_r = cv2.equalizeHist(r)
    # cv2.imshow('split pupil red', equalized_img_r)
    # equalized_img_g = cv2.equalizeHist(g)
    # cv2.imshow('split pupil green', equalized_img_g)
    # equalized_img_b = cv2.equalizeHist(b)
    # cv2.imshow('split pupil blue', equalized_img_b)
    ret, threshr = cv2.threshold(equalized_img_r, 25, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh pupil', threshr)
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(threshr, kernel, iterations=1)
    # cv2.imshow('eorded', img_dilation) 
    
    maskpupil = np.zeros(img_dilation.shape, dtype=np.uint8)
    cv2.circle(maskpupil, (iris_center_x, iris_center_y), iris_radius -130,  255, thickness = -1) # Create a mask to hide everything else except iris
    pupil_masked = cv2.bitwise_and(img_dilation, maskpupil)
    # cv2.imshow('iris and pupil', pupil_masked)
    
    # kernel2 = np.ones((5,5), np.uint8)
    # img_dilation = cv2.erode(pupil_masked, kernel2, iterations=2)
    # cv2.imshow('dilated', img_dilation) 
    
    # Find contours to find the frame of the image. inner boundary of the black frame in the images.
    # contour_dilated_pupil, _ = cv2.findContours(img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pupiliary_boundary_points = None
    newpupil_boundary_points = []
    
        # Find contours to find the frame of the image. inner boundary of the black frame in the images.
    contours_new, _ = cv2.findContours(pupil_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the contour with the largest area (outer contour)
        new_largest_contour = max(contours_new, key=cv2.contourArea)
        
        # Draw the contour (boundary) on a blank image
        pupiliary_boundary_image = np.zeros_like(pupil_masked)
        cv2.drawContours(pupiliary_boundary_image, [new_largest_contour], -1, 255, thickness=1)  # Draw the largest contour
        
        # # Display the boundary image (white boundary on black background)
        # cv2.imshow('Boundary Image', pupiliary_boundary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Extract boundary points from the largest contour
        pupiliary_boundary_points = new_largest_contour.squeeze(axis=1)  # Remove redundant axis
        # print("Boundary Points:")
        # print(boundary_points) 
    
    # print ('pupil boundary',pupiliary_boundary_points)
    inverted = cv2.bitwise_not(pupil_masked)
    # cv2.imshow('inverted pupil', inverted)
    newpupil_boundary_points = uh.traverse_towards_center(inverted, pupiliary_boundary_points, image_center_x, image_center_y, error = 5) #find iris_boundary points
    newpupil_boundary_array = np.array(newpupil_boundary_points) #Convert to numpy array
    
    # print ('pupil array',newpupil_boundary_array)
    #find iris center and draw iris circle
    
    # Find the center of the iris.
    newpupil_center_x, newpupil_center_y, newpupil_radius = uh.fit_circle_to_points(newpupil_boundary_array)
    newpupil_radius  = int(newpupil_radius) 

    # print ('iris circle:',iris_center_x,', ',iris_center_y,', ',iris_radius)
    newImage= cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.circle(newImage, (newpupil_center_x, newpupil_center_y), newpupil_radius, (0,255,0), thickness = 2)

    
    cv2.circle(newImage, (newpupil_center_x, newpupil_center_y), 3, (0,0,255), thickness = -1)
    cv2.circle(newImage, (iris_center_x, iris_center_y), 3, (0, 255, 0), -1)  # Red center point
    # print('shape :', newImage.shape)
    # print('shape/2: ', newImage.shape[0]/2, ' ', newImage.shape[1]/2)
    cv2.circle(newImage, ( int(newImage.shape[1]/2), int(newImage.shape[0]/2)), 3, (255, 0, 0), -1)  # Red center point
    # cv2.waitKey(0)
    # cv2.imshow('pupil_circle', newImage)    
###################################################################################################################################
    # gray_img = cv2.cvtColor(iris_image, cv2.COLOR_BGR2GRAY)
    # # equalized_img = cv2.equalizeHist(gray_img)
    # # cv2.imshow('equal hist', equalized_img)
    # # cv2.waitKey(0)          
    

    # kernel = np.ones((3,3), np.uint8)
    # img_dilation = cv2.erode(gray_img, kernel, iterations=1)
    # # cv2.imshow('Dilated', img_dilation) 
    # # cv2.waitKey(0)        

    # ret, thresh1 = cv2.threshold(img_dilation, 240, 255, cv2.THRESH_BINARY_INV) 
    # # cv2.imshow('threshsed',thresh1)
    # # cv2.waitKey(0)
        
    # img_dilation = cv2.erode(thresh1, kernel, iterations=3)
    # # cv2.imshow('Dilated2', img_dilation) 
    # # cv2.waitKey(0) 
        
    # contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # zzz= cv2.cvtColor(img_dilation,cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(zzz, contours, -1, (0, 255, 0), 2)  # -1 means draw all contours
    
    # # cv2.imshow('contours', zzz)
    # # cv2.waitKey(0)
    
    # # find the cirlce of the flash/ light in the image. In all images light reflection is inside the pupil
    # flash_center_x = 0
    # flash_center_y = 0
    # flash_radius = 0
    # flash_points = None
    # pupil_boundary_points = None
    
    # ##################################################  pupil Boundary #################################
    # # Ensure at least one contour is found. select the 2nd largest contour. 
    # if len(contours) > 0:
    #     # Get the contour with the largest area (outer contour)
    #     # Sort contours by area in descending order
    #     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #     if len(sorted_contours) >= 2:
    #         # Get the second largest contour (index 1 in zero-based indexing)
    #         second_largest_contour = sorted_contours[1]
                
        
    #     # Draw the contour (boundary) on a blank image
    #     boundary_image = np.zeros_like(img)
    #     cv2.drawContours(boundary_image, [sorted_contours[1]], -1, 255, thickness=1)  # Draw the largest contour
        
    #     # Display the boundary image (white boundary on black background)
    #     # cv2.imshow('Boundary Image', boundary_image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     # Extract boundary points from the largest contour
    #     flash_points = sorted_contours[1].squeeze(axis=1)  # Remove redundant axis
        
    #     # print("Boundary Points:")
    #     # print(boundary_points) 

        
    #         # Fit a circle to the points
    #     flash_center_x, flash_center_y, flash_radius = uh.fit_circle_to_points(flash_points)
    #     flash_radius += 65 

    #     image= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        
    #     # cv2.circle(iris_image, (flash_center_x, flash_center_y), flash_radius, (0,255,0), thickness = 2)
    #     # cv2.circle(iris_image, (flash_center_x, flash_center_y), 3, (0, 255, 0), -1)  # Red center point
        
    #     # cv2.imshow('flash', iris_image)
    #     # cv2.waitKey(0)
    
    # else:
    #     print("No pupil contour found in the mask image.")    
        
    # Find the normalized image.
    image_nor = uh.daugman_normalization(color_image, 64, 512,newpupil_center_x, newpupil_center_y, newpupil_radius, iris_radius- newpupil_radius)
    
    
    

    def enhance_iris_strip(image, block_size=8):
        # Read the input image
        
        if image is None:
            raise ValueError("Image not found or unable to read image")

        # Get image dimensions
        height, width = image.shape
        
        # Create an empty image to store the estimated illumination
        estimated_illumination = np.zeros_like(image, dtype=np.float32)
        
        # Loop through each block and calculate the mean
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Define the block region
                block = image[y:y + block_size, x:x + block_size]
                
                # Calculate the mean of the block
                block_mean = np.mean(block)
                
                # Assign the mean value to the corresponding block in the estimated illumination image
                estimated_illumination[y:y + block_size, x:x + block_size] = block_mean
        
        # Subtract the estimated illumination from the original image
        uniformly_illuminated_image = image.astype(np.float32) - estimated_illumination
        
        # Normalize the image to the range [0, 255]
        uniformly_illuminated_image = cv2.normalize(uniformly_illuminated_image, None, 0, 255, cv2.NORM_MINMAX)
        uniformly_illuminated_image = uniformly_illuminated_image.astype(np.uint8)
        
        # Apply histogram equalization to enhance the contrast
        enhanced_image = exposure.equalize_hist(uniformly_illuminated_image)
        
        # Convert the image back to uint8 format after histogram equalization
        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        
        return uniformly_illuminated_image, enhanced_image

    
    
    nor_gray = cv2.cvtColor(image_nor, cv2.COLOR_BGR2GRAY)
    
    ui, equal_hist = enhance_iris_strip(nor_gray)

    # cv2.imshow('normalized image',equal_hist)
    # cv2.imshow('uniformally_illuminated_image', ui)
    #################################################Haar wavelet ################################################
    def haar_wavelet_decomposition(image, levels=2):            
        if image is None:
            raise ValueError("Image not found or unable to read image")

        # Normalize the image to float32
        image = image.astype(np.float32)

        # Perform Haar wavelet decomposition
        coeffs = image
        for level in range(levels):
            coeffs = pywt.dwt2(coeffs, 'haar')
            LL, (LH, HL, HH) = coeffs
            coeffs = LL  # Only keep the LL part for the next level of decomposition

        # Normalize the final LL image to the range [0, 255] for visualization
        LL_normalized = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
        LL_normalized = LL_normalized.astype(np.uint8)

        return LL_normalized
    
    reduced_image = haar_wavelet_decomposition( nor_gray )
    # cv2.imshow('Haar Reduced Image', reduced_image)
    # print('reduced image size: ', reduced_image.shape)

    

    ############################################## Apply the gabor filters ########################################

    # Load normalized iris image (grayscale)  
    # himg1 = equal_hist[:,0:64]
    # himg2 = equal_hist[:,229:361]
    # iris_image = cv2.hconcat([himg1, himg2])
    # cv2.imshow('concat image1',iris_image)

    # iris_image = equal_hist

    def generate_gabor_wavelet(size, freq, angle):
        # Create 2D grid
        x, y = np.meshgrid(np.arange(-size//2, size//2), np.arange(-size//2, size//2))

        # Rotation
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)

        # Gabor wavelet formula
        wavelet_real = np.exp(-(x_rot**2 + y_rot**2) / (2 * (size / 3)**2)) * np.cos(2 * np.pi * freq * x_rot)
        wavelet_imag = np.exp(-(x_rot**2 + y_rot**2) / (2 * (size / 3)**2)) * np.sin(2 * np.pi * freq * x_rot)

        return wavelet_real, wavelet_imag
    
    def encode_iris_pattern(iris_image):
        # Define parameters for Gabor wavelets
        wavelet_size = 9                    #[7, 9, 15]  # Different wavelet sizes
        wavelet_frequency = 0.2             #][0.1, 0.2, 0.3]  # Different wavelet frequencies
        wavelet_orientation = np.pi/4       # [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Different wavelet orientations

        # Initialize empty bit string
        encoded_bits = ""


        wavelet_real, wavelet_imag = generate_gabor_wavelet(wavelet_size, wavelet_frequency, wavelet_orientation)

        # Convolve iris image with real and imaginary parts of the wavelet
        response_real = convolve2d(iris_image, wavelet_real, mode='same', boundary='wrap')
        response_imag = convolve2d(iris_image, wavelet_imag, mode='same', boundary='wrap')
        # cv2.imshow('convolve real', response_real)
        # cv2.imshow('convolve imaginary', response_imag)

        
        for rows in range (response_real.shape[0]):
            for col in range(response_real.shape[1]):
            
                if response_real[rows, col] >= 0:
                    encoded_bits += "1"
                else:
                    encoded_bits += "0"
                
                if response_imag[rows, col] >= 0:
                    encoded_bits += "1"
                else:
                    encoded_bits += "0"
                        

        return encoded_bits

    # def quantize_angle(angle):
    #     # Quantize angle into 4 quadrants
    #     quantized_angle = ""
    #     for a in angle.flatten():
    #         if -np.pi/4 <= a < np.pi/4:
    #             quantized_angle += "00"
    #         elif np.pi/4 <= a < 3*np.pi/4:
    #             quantized_angle += "01"
    #         elif -3*np.pi/4 <= a < -np.pi/4:
    #             quantized_angle += "10"
    #         else:
    #             quantized_angle += "11"
    #     return quantized_angle

    encoded_bits = encode_iris_pattern(reduced_image)
    # print("Encoded bits:", len(encoded_bits))
    # print('binary_iris_code: ', encoded_bits)
    
    # encoded_bits_part1 = encoded_bits[0:2048]
    # encoded_bits_part2 = encoded_bits[2048:]
    encoded_list = [int(bit) for bit in encoded_bits]
    # print ('encoded lils:::::: ', encoded_list)
        
    # hd=0
    # for i in range(0, len(encoded_bits)):
    #     if encoded_bits[i]=='1':
    #         hd+=1
        
            
            
    # print('hd: ',hd )  
            
    # print('final_code_part1:',iris_int_code1)
    # print('final_code_part2:',iris_int_code2)



    # # Define Gabor filter parameters
    # frequencies = [0.1, 0.2, 0.3]  # Gabor filter frequencies
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Gabor filter angles

    # # Apply Gabor filters
    # filtered_responses = uh.apply_gabor_filters(iris_image, frequencies, angles)

    # # Encode iris features into a 2048-bit binary iris code
    # iris_code = uh.encode_iris_features(filtered_responses, num_bits=2048)

    # # Print or use the iris code as needed
    # # np.set_printoptions(threshold=np.inf)
    # # print("Encoded Iris Code (2048 bits):", iris_code)
    
#################################################################################################################################################3        
    
    # if count == 0:
    #     iris_code_1 = iris_code.tolist()
    #     print('iris code 1: ', iris_code_1)
    # if count == 1:
    #     iris_code_2 = iris_code.tolist()
    #     print('iris code 2: ', iris_code_2)
        
    # if count == 0:
    #     count +=1    
        
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
    else :
        cv2.destroyAllWindows() 
        
        
    return encoded_list           
                
    # Perform XOR operation
    # result = np.bitwise_xor(iris_code_1, iris_code_2) 

    # Using list comprehension
    # result = [a ^ b for a, b in zip(iris_code_1, iris_code_2)]   

    # # Print the XOR result
    # np.set_printoptions(threshold=np.inf)
    # print("XOR for left_001_1&2:", result)

    # iris1_int = conv.binary_array_to_integers(iris_code_1)
    # iris2_int = conv.binary_array_to_integers(iris_code_2)


    # def hamweight(result):
    #     hamming = 0
    #     for bit in result:
    #         # Increment the counter if the bit is 1
    #         if bit == 1:
    #             hamming += 1
    #     return hamming        
                
    # # Return the count of 1s
    # print('XOR', hamweight(result))
    # print('iris1', hamweight(iris_code_1))
    # print('iris2', hamweight(iris_code_2))
            

