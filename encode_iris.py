import cv2 
import numpy as np
import utils_preprocess as uh

#encode
'''
This function takes iris image (UPOL dataset images) and
returns the 4096 bits of iris data code.

STEPS:
1. Read the image
2. Find the inner frame boundary of the iris image(inner boundary of the black frame in the image).
3. Perform processing on image to make iris distinguishingly visible for the algorithm
4. Find the Iris' circular boundary(using utility functions from utils_preprocess).
5. Find the pupil's circular boundary(using utility functions from utils_preprocess).
6. Apply daugman's normalization to obtain rectangular image(scale and pupil dilation invariant) of iris.
7. Apply haar wavelet decomposition to reduced the size of the image.
8. Encode the reduced normalized image using the gabor wavelets.
'''
def encode(im_path):
    ''' Step 1. Read the image '''
    image = im_path
    color_image = cv2.imread(image) # read color image
    img = cv2.imread(image,0)   # read gray image
    
    '''Step 2. Find the inner frame boundary of the iris image(inner boundary of the black frame in the image).'''    
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

    '''Step 3. Perform processing on image to make iris distinguishingly visible for the algorithm'''
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
    '''Step 4. Find the Iris' circular boundary(using utility functions from utils_preprocess).'''
    iris_boundary_points = uh.traverse_towards_center(Gaussian, boundary_points, image_center_x, image_center_y, error = 25) #find iris_boundary points
    iris_boundary_array = np.array(iris_boundary_points) #Convert to numpy array
    
    #find iris center and draw iris circle
    
    # Find the center of the iris.
    iris_center_x, iris_center_y, iris_radius = uh.fit_circle_to_points(iris_boundary_array)
    iris_radius  = int(iris_radius * 0.9) 

    # newImage= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # cv2.circle(color_image, (iris_center_x, iris_center_y), iris_radius, (0,255,0), thickness = 2)
    # cv2.imshow('iris_circle', color_image)
    # cv2.waitKey(0)

    mask = np.zeros(color_image.shape, dtype=np.uint8)
    cv2.circle(mask, (iris_center_x, iris_center_y), iris_radius, (255, 255, 255), thickness = -1) # Create a mask to hide everything else except iris
    iris_image = cv2.bitwise_and(color_image, mask)
    # cv2.imshow('iris and pupil', iris_image)
    # cv2.circle(iris_image, (iris_center_x, iris_center_y), 3, (255, 0, 0), thickness = -1)
    
#################################### Preprocess before finding pupil boundary ##################################################     
      
    b,g,r = cv2.split(iris_image)
    equalized_img_r = cv2.equalizeHist(r)
    ret, threshr = cv2.threshold(equalized_img_r, 25, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh pupil', threshr)
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(threshr, kernel, iterations=1)
    # cv2.imshow('eorded', img_dilation) 
    
    maskpupil = np.zeros(img_dilation.shape, dtype=np.uint8)
    cv2.circle(maskpupil, (iris_center_x, iris_center_y), iris_radius -130,  255, thickness = -1) # Create a mask to hide everything else except iris
    pupil_masked = cv2.bitwise_and(img_dilation, maskpupil)
    # cv2.imshow('iris and pupil', pupil_masked)
    
    '''Step 5. Find the pupil's circular boundary(using utility functions from utils_preprocess).'''
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
    
    inverted = cv2.bitwise_not(pupil_masked)
    # cv2.imshow('inverted pupil', inverted)
    newpupil_boundary_points = uh.traverse_towards_center(inverted, pupiliary_boundary_points, image_center_x, image_center_y, error = 5) #find iris_boundary points
    newpupil_boundary_array = np.array(newpupil_boundary_points) #Convert to numpy array
      
    # Find the center of the iris.
    newpupil_center_x, newpupil_center_y, newpupil_radius = uh.fit_circle_to_points(newpupil_boundary_array)
    newpupil_radius  = int(newpupil_radius) 

    # print ('iris circle:',iris_center_x,', ',iris_center_y,', ',iris_radius)
    newImage= cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.circle(newImage, (newpupil_center_x, newpupil_center_y), newpupil_radius, (0,255,0), thickness = 2)
    
    cv2.circle(newImage, (newpupil_center_x, newpupil_center_y), 3, (0,0,255), thickness = -1)
    cv2.circle(newImage, (iris_center_x, iris_center_y), 3, (0, 255, 0), -1)  # Red center point
    cv2.circle(newImage, ( int(newImage.shape[1]/2), int(newImage.shape[0]/2)), 3, (255, 0, 0), -1)  # Red center point
    # cv2.waitKey(0)
    # cv2.imshow('pupil_circle', newImage)    
    
    '''Step 6. Apply daugman's normalization to obtain rectangular image(scale and pupil dilation invariant) of iris.'''    
    # Find the normalized image.
    image_nor = uh.daugman_normalization(color_image, 64, 512,newpupil_center_x, newpupil_center_y, newpupil_radius, iris_radius- newpupil_radius)
        
    nor_gray = cv2.cvtColor(image_nor, cv2.COLOR_BGR2GRAY)
    

    
    '''Step 7. Apply haar wavelet decomposition to reduced the size of the image.'''
    reduced_image = uh.haar_wavelet_decomposition( nor_gray )
    # cv2.imshow('Haar Reduced Image', reduced_image)
    # print('reduced image size: ', reduced_image.shape)
    


    '''Step 8. Encode the reduced normalized image using the gabor wavelets.'''
    encoded_bits = uh.encode_iris_pattern(reduced_image)
    # print("Encoded bits:", len(encoded_bits))
    # print('binary_iris_code: ', encoded_bits)
    
    encoded_list = [int(bit) for bit in encoded_bits]
    # print ('encoded list:', encoded_list)
        
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
    else :
        cv2.destroyAllWindows() 
        
        
    return encoded_list