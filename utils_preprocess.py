import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve2d
import cv2
import pywt # python wavelets


####################################################################################################################
# fit_circle_to_points: for the given "points" find the possible circle. 
# It returns the center of the circle i.e. x and y  and it also returns 
# the radius of the corresponding circle.
####################################################################################################################
def fit_circle_to_points(points):
    # Define the objective function to minimize (sum of squared errors)
    def objective(params):
        x_center, y_center, radius = params
        distances = np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2)
        return np.sum((distances - radius)**2)
    
    # Initial guess for center and radius (centroid of points 
    #   and average distance to centroid)
    x0 = np.mean(points[:, 0])
    y0 = np.mean(points[:, 1])
    r0 = np.mean(np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2))
    
    # Minimize the objective function to find the optimal center and radius
    result = minimize(objective, [x0, y0, r0], method='Nelder-Mead')
    
    # Extract the optimized parameters (center coordinates and radius)
    x_center, y_center, radius = result.x
    # print(f'fit_circle_to_points:{x_center}, {y_center}, {radius} ')
    return int(x_center), int(y_center), int(radius)


######################################################################################################################
#traverse towards center:  from the above found inner perimeter of the black frame traverse in the direction of
#                          the center. find the white pixel in the thresholded image. if the distance between the 
#                          white pixel and the frame boundary is less than 25 pixel do not add it to the iris boundary
#                          pixel array.
######################################################################################################################
def traverse_towards_center(image, points, image_center_x, image_center_y, error):
    height, width = image.shape[:2]
    iris_boundary_points = []
    
    # # For visualization of the found points for circle.
    # image2 = np.copy(image)
    # image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)    

    # Iterate over each point in the array    
    for point in points:
        x, y = point
        # Determine the direction vector towards the center point C
        dx = image_center_x - x
        dy = image_center_y - y
              
        # Calculate the distance to the center
        distance_to_center = np.sqrt(dx**2 + dy**2)
        
        # Normalize the direction vector
        if distance_to_center > 0:
            dx /= distance_to_center
            dy /= distance_to_center
        
        # Interpolate from current point towards the center point
        current_x, current_y = x, y
        while True:
            # Round to nearest integer coordinates
            px, py = int(round(current_x)), int(round(current_y))
            
            # Check if the pixel coordinates are within image bounds
            if 0 <= px < width and 0 <= py < height:
                # Access and process the pixel 
                if image[py, px] != 0: # Check pixel value
                    distance = np.sqrt((py - y)**2 + (px - x)**2) # Find pixel's distance from the center
                    if int(distance) > error:
                        iris_boundary_points.append([int(px), int(py)]) # Add pixel to iris boundary points array
                        # image2[py, px] = [0,255,0] #draw points found on the image
                    break
                    
                
            # # Move towards the center by a small step
            current_x += dx
            current_y += dy                          
            
            # Break loop if reached or moved past the center point
            if np.sqrt((current_x - image_center_x)**2 + (current_y - image_center_y)**2) <= 1.0:
                break
    
    # cv2.imshow('found_circle', image2)
    # cv2.waitKey(0)
    return iris_boundary_points



###########################################################################################################
# daugman_normalization: To convert the concentric iris image into a rectangular image. This is the 
#                        liang's code for daugman normalization. r_out is the distance between 
#                        the pupil circle and the iris circle
###########################################################################################################
# liang's code 
def daugman_normalization(image, height, width, newpupil_center_x, newpupil_center_y, r_in, r_out):       
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out #r_out is changed to the radius of the outer circle
    
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    
    circle_x = int(newpupil_center_x)
    circle_y = int(newpupil_center_y)

    '''
    Here now for each pixel in the empty flatten image is assigned a pixel value 
    from the concentric iris image as per daugman's rubber sheet model.
    '''
    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            # color of the pixel
            color = image[int(Yc)][int(Xc)]  #image column is height and row is width

            flat[j][i] = color
    return flat   

###########################################################################################################
# Haar wavelet Decomposition: To reduced the size of the image without losing the texture info.
###########################################################################################################

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

###############################################################################################
# generate_gabor_wavelet : generate the gabor wavelet of the required parameters(experimented)  
#                          as per the gabor wavelet equation in the daugman's paper.
############################################################################################### 
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

################################################################################################
# encode_iris_pattern: This function applies the gabor wavelet to the reduced normalized image of
#                      iris. It convolves the real and imaginary wavelet part with the image and
#                      and then bit values are assigned as per the phase quantization mentioned
#                      in the paper.
################################################################################################
def encode_iris_pattern(iris_image):
    # Define parameters for Gabor wavelets
    wavelet_size = 9                    #[7, 9, 15]  # Different wavelet sizes
    wavelet_frequency = 0.2             #][0.1, 0.2, 0.3]  # Different wavelet frequencies
    wavelet_orientation = np.pi/4       # [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Different wavelet orientations

    # Initialize empty bit string to store the iris code
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







'''
Following code is currently not in use. Preserved for future Usage depending on the requirement.
'''
# #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ###################################################################################################
# # apply gabor filters: apply the gabor filters on image for different angles.
# ###################################################################################################
# def apply_gabor_filters(image, frequencies, angles):
#     filtered_responses = []
#     gabor = 1
#     for freq in frequencies:
#         for angle in angles:
#             # Create Gabor kernel
#             kernel = cv2.getGaborKernel((21, 21), freq, angle, 21, 0.8, 0, ktype=cv2.CV_32F)
#             # cv2.imshow('gabor kernel',kernel)
            
#             # Apply Gabor filter to the image
#             filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
#             i_name = f'gabor{gabor}.png'
#             cv2.imwrite(i_name, filtered_image)
#             gabor +=1
            
            
#             # Append filtered image to the list
#             filtered_responses.append(filtered_image)
#     # cv2.imshow('gabor',filtered_image)
#     # cv2.waitKey(0)
#     return filtered_responses


# ##################################################################################################
# # encode iris features : convert the filtered responses of the gabor filter into 2048 bits code
# #                        based on the phase value.
# ##################################################################################################
# def encode_iris_features(filtered_responses, num_bits=2048):
#     # Resize filtered responses into a 1D array
#     feature_vector = np.array([])
#     for response in filtered_responses:
#         feature_vector = np.concatenate((feature_vector, response.flatten()))
    
#     # Perform quantization and thresholding to generate binary iris code
#     threshold = np.median(feature_vector)
#     iris_code = (feature_vector > threshold).astype(int)
    
#     # Resize iris code to specified number of bits
#     if len(iris_code) < num_bits:
#         iris_code = np.concatenate((iris_code, np.zeros(num_bits - len(iris_code))))
#     elif len(iris_code) > num_bits:
#         iris_code = iris_code[:num_bits]
    
#     return iris_code

# def gabor_filter_bank(ksize, sigma, theta, lambd, gamma):
#     filters = []
#     # gabor = 1
#     for t in theta:
#         params = {'ksize': (ksize, ksize), 'sigma': sigma, 'theta': t, 'lambd': lambd, 'gamma': gamma}
#         kernel = cv2.getGaborKernel(**params, ktype=cv2.CV_64F)
#         # i_name = f'Kernel{gabor}.png'
#         # cv2.imwrite(i_name, kernel)
#         # gabor +=1        
#         filters.append(kernel)
#     return filters

# def extract_iris_code(image, filters):
#     features = []
#     gabor = 1
#     for filter in filters:
#         filtered_image = cv2.filter2D(image, cv2.CV_64F, filter)
#         i_name = f'gabor{gabor}.png'
#         cv2.imwrite(i_name, filtered_image)
#         gabor +=1
#         feature = np.mean(filtered_image)
#         features.append(feature)
#     return features