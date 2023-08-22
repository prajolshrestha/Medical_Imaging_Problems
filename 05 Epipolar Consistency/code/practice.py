import numpy as np
import cv2

from skimage.transform import radon, rescale

# Goal: Implementation of the epipolar consistency measure as
#       Described in "Efficient Epipolar Consistency" by Aichert et al. (2016)

def main():

    shape = [1920, 2480]
    scale_computation = 0.05
    scales_show = (0.4, 4)

    # Load all the data from the disk
    projections_original = load_projections('../P5_epipolar_consistency/data/ProjectionImage', shape)
    projection_matrices = load_geometry('../P5_epipolar_consistency/data/ProjectionMatrix')

    # Downscale the projection images
    projections = scale_images(projections_original, scale_computation)
    
    # Compute the derivatives on the sinograms (derivative of Radon transform in each projection)
    radon_derivatives = compute_radon_derivatives(projections)
    
    K0, K1 = compute_mapping_circle_to_epipolar_lines(projection_matrices[0],
                                                     projection_matrices[1], shape)
    

    consistency = compute_constistency(radon_derivatives,[K0, K1])



    # Display
    show_projections(projections_original, scales_show[0])
    show_radon_derivatives(radon_derivatives, scales_show[1])

    if np.abs(consistency - 14.61544) < 1e-2:
        print('Your mapping is correct')
    else:
        print('The mapping still doesn\'t work!')




################################################### Computaion Functions ##################################    
# load raw image data 
def load_projections(projection_filename, shape):

    projections = [] # each list entry is one numpy array

    # use projection_filename and load all (two) enumerated files from data subdirectory
    for i in range(2):
        filename = f"{projection_filename}{i}.npy" # F-strings used (formatted string literals)
        projection = np.load(filename)
        projection = projection.reshape(tuple(shape))
        #print(projection.shape)
        #print(type(temp))

        projections.append(projection)

    # make sure the projections are returned with correct dimensions
    for i in range(len(projections)):
        if projections[i].shape != tuple(shape):
            raise ValueError(f"Projection {i} does not have the correct dimension")

    #print(len(projections))
    return projections


# Load the raw data for the projection matrices
def load_geometry(matrices_filename):

    projection_matrices = [] # each list is one Numpy array

    # use matrices_filename and load all (two) enumerated files from data subdirectory
    for i in range(2):
        filename = f"{matrices_filename}{i}.npy"
        matrix = np.load(filename) # load numpy array from the file
        #print(matrix.shape) # ==> (12,)
        matrix = matrix.reshape((3,4))
        projection_matrices.append(matrix)

    # make sure that the projection matrices are returned with correct dimensions
    for i in range(len(projection_matrices)):
        if projection_matrices[i].shape != (3,4):
            raise ValueError(f"Matrix {i} does not have the correct shape.")
        

    return projection_matrices
  
# Scale each entry of a list of images
def scale_images(images, scale=1.0):
    
    images_scaled = []
    for image in images:
        # Scale each image
        images_scaled.append(rescale(image, scale=scale, mode='reflect', anti_aliasing=False))

    return images_scaled


# Compute Radon Derivatives
def compute_radon_derivatives(projections):
    
    # Generate an array of angles from 0 to 180 degress.
    
    # Number of angles is determined based on the maximum dimension of the projections.
    # np.sqrt(2) is used as a factor to determine the number of angles. 
    # This choice is based on a common practice in image processing and tomography. 
    # The idea is to sample the angular space at a rate that is greater than or equal to the Nyquist rate, 
    # which is twice the highest frequency component in the data. 
    # In the context of Radon transforms, this means that you should sample angles at a rate that captures the diagonal information in the image.
    # Consider a square image. The diagonal of the square is longer than its sides, and it represents the highest spatial frequency in the image. 
    # By using np.sqrt(2) as a factor, you are ensuring that you sample angles at a rate that can capture the information along the diagonals of the image
    # Thus, it ensures a sufficient angular resolution to capture diagonal information in the image.

    # endpoint = False means endpoint '180 degree' is not included.
    theta = np.linspace(0., 180., int(np.ceil(max(projections[0].shape) * np.sqrt(2))), endpoint=False)

    radon_derivatives = [] #To store computed radon derivatives

    for projection in projections:

        # Sinogram
        # Compute Radon Transform 'Sinogram' of the current projection using specified  'theta' values.
        # input = False means input image is not considered as a circle
        sinogram = radon(projection, theta=theta, circle=False)

        # Compute 'gradient of sinogram' along horizontal axis (axis = 1)
        radon_derivative = np.gradient(sinogram,axis=1)
        radon_derivatives.append(radon_derivative)

    return radon_derivatives
        

def compute_mapping_circle_to_epipolar_lines(p0, p1, shape):
    
    # Calculate center of projection from projection matrix
    C0 = compute_projection_center(p0)
    C1 = compute_projection_center(p1)
    print(C0.shape)

    # Plücker coordinates of the baseline using two camera centers
    B = get_pluecker_coordinates(C0,C1)

    # mapping from [cos(kappa) sin(kappa)]' to epipolar plane
    K = compute_mapping_circle_to_plane(B)

    # For each projection
    K0 = compute_mapping_per_projection(p0, K, shape)
    K1 = compute_mapping_per_projection(p1, K, shape)

    return K0, K1


# Calculate Center of Projection 
def compute_projection_center(pm):

    # Consider P = [M, p4] where p4 is simply the fourth column vector
    # and solve for ker(P) by -M^{-1}*p4
    M = pm[:, :3]
    p4 = pm[:, 3]

    # Calculate the camera center in homogenious coordinates
    C = -np.linalg.inv(M).dot(p4)

    # Convert to 4-elements vector in homogeneous coordinates
    C = np.concatenate([C, [1]]) # 4D vector banako
    print(C)

    return C


# Compute the pluecker coordinates 
#
# Plücker coordinates are a mathematical representation used in geometry and physics to describe a line in three-dimensional space
# Represents line's Properties: such as, its  position and direction, without explictly specifying two points on the line
# - A concise way to represent a line's essential characteristics
# - Where the line is and in which direction it's headed.


# Moment Vector (m) represents location or position of line in space(A point in 3D space)
# Direction vector (d) reprsents line's direction. It tells us which way the line is pointing in 3D space. (Arrow indicating line's path)
def get_pluecker_coordinates(C0, C1):

    # Compute "direction vector (d)"" between the two camera centers C0 and C1
    d = C1[:3] - C0[:3]
    d_normalized = d / np.linalg.norm(d) # Normalize the direction vector

    
    # Compute the "moment vector(m)" by taking the cross product of C0 with d_normalized
    m = np.cross(C0[:3],d_normalized)

    # Calculate "pluecker coordinates" based on the moment and direction vectors
    B = (m[0], m[1], m[2], d_normalized[0], d_normalized[1], d_normalized[2]) 


    return B

# Compute the mapping as described in the article
def compute_mapping_circle_to_plane(B):

    K = np.zeros([4, 2])

    a2, s2 = compute_pluecker_base_moment(B)
    a3, s3 = compute_pluecker_base_direction(B)

    # Compute the mapping K as described in section III.C of the article
    K[0, 0] = a3[0] / s3
    K[0, 1] = a3[1] / s3
    K[1, 0] = -a2[0] / s2
    K[1, 1] = -a2[1] / s2
    K[2, 0] = -a2[2] / s2
    K[2, 1] = a3[2] / s3
    K[3, 0] = a3[2] / s3
    K[3, 1] = -a2[2] / s2

    return K



# Plucker line moment
def compute_pluecker_base_moment(B):

    a2 = np.array([B[3], -B[1], B[0]])
    s2 = np.linalg.norm(a2)

    return a2, s2

# Plucker  line direction
def compute_pluecker_base_direction(B):
    
    a3 = np.array([-B[2], -B[4], -B[5]])
    s3 = np.linalg.norm(a3)

    return a3, s3


# Compute individual mappings for a specific projection
def compute_mapping_per_projection(p, K, shape):
    """
    Compute the individual mapping for a specific projection.

    Args:
        p: Projection matrix.
        K: Transformation matrix.
        shape: Shape of the image.

    Returns:
        Kp: Resulting mapping matrix.
    """
    # Step 1: Calculate the Pseudo inverse of projection matrix p
    p_pseudo_inv = np.linalg.pinv(p) # psudoinverse

    # Calculate homography matrix H as described in the refrence
    nx, ny = shape[0], shape[1]
    H = np.array([[1, 0, 0], 
                  [0, 1, 0],
                  [nx / 2, ny / 2, 1]])
    
    #print(H.T.shape)
    # Lines transform contra-variant by their inverse transpose
    # Step 2: Apply transformation representated by K to the pseudoinverse of p
    Kp = np.dot(H, np.dot(p_pseudo_inv.T, K))

    return Kp


# Compute _consistency
def compute_constistency(radon_derivatives, K):

    consistency = 0
    # iterate from -190 to 179 (360 degrees in total) angles
    for i in range(-180, 180):

        # Calculate x_kappa, a 2D vector representing the unit vector
        # in the direction of angle kappa 
        # And append its dot product with K to 'ls' list.
        ls = []
        kappa = i*0.0035
        x_kappa = np.array([np.cos(kappa), np.sin(kappa)])
        ls.append(np.dot(K[0], x_kappa))
        ls.append(np.dot(K[1], x_kappa))

        values = []
        list_radon_domain = [] # to store tuple (alpha, t) representing values related to radon domain
        dtrs = [] # to store tuples (dtr_x, dtr_y) representing normalized coordinates in Radon derivative space
        for l, dtr in zip(ls, radon_derivatives): # iterate over pair of elements

            length = np.linalg.norm(l[0:2]) # compute length 
            alpha = np.arctan2(l[1], l[0]) # compute arctangent
            t = l[2] / length 
            list_radon_domain.append((alpha,t))

            # Scale 
            dtr_x = alpha / np.pi + 1 # scale to 0 - 2 range
            dtr_y = t / 3164 + 0.5 # scale to 0 - 1 range
            dtrs.append((dtr_x, dtr_y))

            # Compute normalized coordinates in Radon derivative
            # also accounts for symmetry rho(alpha,t) = -rho(alpha+pi, -t)
            if dtr_x > 1:
                weight = -1
                dtr_x = dtr_x - 1.0 # adjust to account for symmetry
                dtr_y = 1.0 - dtr_y # adjust to account for symmetry
            else:
                weight =  1 

            # Scale dtr_x & dtr_y to the dimensions of the Radon derivative array 'dtr'
            dtr_x = (1 - dtr_x) * dtr.shape[0]
            dtr_y = (1 - dtr_y) * dtr.shape[1]

            # Nearest neighbor interpolation to find value at the scaled coordinates in dtr and * by weight
            values.append(weight * dtr[np.int(np.round(dtr_x)), np.int(np.round(dtr_y))])
        
        # Compute squared difference between two values
        consistency += np.square(values[0] - values[1])
    
    return consistency


################################### Display Methods #########################################


def show_projections(projections, scale = 1.0):

    projections_scaled = scale_images(projections, scale)
    
    for i, projection in enumerate(projections_scaled):
        # Scale pixel values to [0,255] and convert to 8-bit integer
        p_min = np.min(projection)
        projection_range_scaled = ((projection - p_min) / (np.max(projection) - p_min) * 255).astype(np.uint8)
        
        # Apply a colormap to scaled projection for visualization
        projection_colored = cv2.applyColorMap(projection_range_scaled,cv2.COLORMAP_BONE)
        
        show_image(projection_colored, 'projection {}'.format(i),False)


def show_radon_derivatives(radon_derivatives, scale=1.0):
    
    radon_derivatives_range_scaled = [] # To store scaled and ranged Radon derivarives 

    for i, radon_derivative in enumerate(radon_derivatives):

        # Scale values to [0,255] and convert to 8-bit integer
        rd_min = np.min(radon_derivative)
        radon_derivative_range_scaled = ((radon_derivative - rd_min) * (np.max(radon_derivative) * rd_min) * 255)
        radon_derivatives_range_scaled.append(radon_derivative_range_scaled.astype(np.uint8))

    radon_derivatives_scaled = scale_images(radon_derivatives_range_scaled, scale)

    for i, radon_derivative in enumerate(radon_derivatives_scaled):
        show_image(radon_derivative,'d rho {}/ dt'.format(i), False)



def show_image(i, t, destroy_windows=True):

    cv2.imshow(t,i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()