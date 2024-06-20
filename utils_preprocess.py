import numpy as np
from scipy.optimize import minimize
import cv2

##############################################################################################################################################
# fit_circle_to_points: for the given "points" find the possible circle. It returns the center of the circle i.e. x and y  and it also returns the radius of the 
#                       corresponding circle.
###############################################################################################################################################
def fit_circle_to_points(points):
    # Define the objective function to minimize (sum of squared errors)
    def objective(params):
        x_center, y_center, radius = params
        distances = np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2)
        return np.sum((distances - radius)**2)
    
    # Initial guess for center and radius (centroid of points and average distance to centroid)
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
    # image2 = np.copy(image)
    # image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)    
    # cv2.imshow('traverese_towards center', image)
    # cv2.waitKey(0)
    # Iterate over each point in the array
    
    for point in points:
        x, y = point
        # Determine the direction vector towards the center point C
        dx = image_center_x - x
        dy = image_center_y - y
        
        # print('point: ', point)
        # print('direc: ',dx, dy)
        
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
                        # image2[py, px] = [0,255,0]
                    break
                    
                
            # # Move towards the center by a small step (adjust step size as needed)
            current_x += dx
            current_y += dy                          
            
            # Break loop if reached or moved past the center point
            if np.sqrt((current_x - image_center_x)**2 + (current_y - image_center_y)**2) <= 1.0:
                break
    # cv2.imshow('found_circle', image2)
    # cv2.waitKey(0)
    return iris_boundary_points



###########################################################################################################
# daugman_normalization: To conver the concentric iris image into a rectangular image. This is the 
#                        liang's code.
###########################################################################################################
# liang's code for reference
def daugman_normalization(image, height, width, newpupil_center_x, newpupil_center_y, r_in, r_out):       
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(newpupil_center_x)
    circle_y = int(newpupil_center_y)

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

            color = image[int(Yc)][int(Xc)]  # color of the pixel

            flat[j][i] = color
    return flat   

###################################################################################################
# apply gabor filters: apply the gabor filters on image for different angles.
###################################################################################################
def apply_gabor_filters(image, frequencies, angles):
    filtered_responses = []
    gabor = 1
    for freq in frequencies:
        for angle in angles:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), freq, angle, 21, 0.8, 0, ktype=cv2.CV_32F)
            # cv2.imshow('gabor kernel',kernel)
            
            # Apply Gabor filter to the image
            filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            i_name = f'gabor{gabor}.png'
            cv2.imwrite(i_name, filtered_image)
            gabor +=1
            
            
            # Append filtered image to the list
            filtered_responses.append(filtered_image)
    # cv2.imshow('gabor',filtered_image)
    # cv2.waitKey(0)
    return filtered_responses


##################################################################################################
# encode iris features : convert the filtered responses of the gabor filter into 2048 bits code
#                        based on the phase value.
##################################################################################################
def encode_iris_features(filtered_responses, num_bits=2048):
    # Resize filtered responses into a 1D array
    feature_vector = np.array([])
    for response in filtered_responses:
        feature_vector = np.concatenate((feature_vector, response.flatten()))
    
    # Perform quantization and thresholding to generate binary iris code
    threshold = np.median(feature_vector)
    iris_code = (feature_vector > threshold).astype(int)
    
    # Resize iris code to specified number of bits
    if len(iris_code) < num_bits:
        iris_code = np.concatenate((iris_code, np.zeros(num_bits - len(iris_code))))
    elif len(iris_code) > num_bits:
        iris_code = iris_code[:num_bits]
    
    return iris_code

def gabor_filter_bank(ksize, sigma, theta, lambd, gamma):
    filters = []
    # gabor = 1
    for t in theta:
        params = {'ksize': (ksize, ksize), 'sigma': sigma, 'theta': t, 'lambd': lambd, 'gamma': gamma}
        kernel = cv2.getGaborKernel(**params, ktype=cv2.CV_64F)
        # i_name = f'Kernel{gabor}.png'
        # cv2.imwrite(i_name, kernel)
        # gabor +=1        
        filters.append(kernel)
    return filters

def extract_iris_code(image, filters):
    features = []
    gabor = 1
    for filter in filters:
        filtered_image = cv2.filter2D(image, cv2.CV_64F, filter)
        i_name = f'gabor{gabor}.png'
        cv2.imwrite(i_name, filtered_image)
        gabor +=1
        feature = np.mean(filtered_image)
        features.append(feature)
    return features