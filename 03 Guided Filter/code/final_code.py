import numpy as np
import cv2

# TASK: Implementation of guided filter as described in the original paper by He et al. (2013)
#
# Step 1:   Implement a box filter function to perform the mean filter with box/window parameter r in 2D.  
#           In the original paper a 1D box filter via moving sum is given (cf. Algorithm 2).
#           Here we expand this for the 2D case by successively applying it for each dimension.
# Step 2:   Implement the guided filter as shown in the paper (Algorithm 1).

def main():

    # Initialization of parameters (tra also other value combinations!)
    r = 8
    epsilon = 0.001# ** 2

    # Load and prepare images
    image_rgb = cv2.imread('../P3_guided_filter/data/coronaries_noisy.jpg')
    guidance_rgb = cv2.imread('../P3_guided_filter/data/coronaries.jpg') #coronaries.jpg
    print(image_rgb.shape)
    # Convert RGB images to gray
    image = convert2gray(image_rgb)
    guidance = convert2gray(guidance_rgb)

    # Example 1: guidance is original image without noise
    filtered_image_1 = guided_filter(image, guidance, r, epsilon)

    # Example 2: guidance is original image with noise
    filtered_image_2 = guided_filter(image, image, r, epsilon)

    # Example 3: guided filter color
    filtered_image_3 = guided_filter_color(image, guidance_rgb,r,epsilon) #Error


    # Show all images
    show_image(image, 'input',False)
    show_image(guidance, 'Guidance', False)
    show_image(filtered_image_1, 'example 1', False)
    show_image(filtered_image_2, 'example 2', False)
    show_image(filtered_image_3, 'example 3', False)



########################## Functions for Calculations #############################
def convert2gray(image):
    temp = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0
    return image_gray

# perform mean filtering for each column
def box_filter_columns(img,r):

    rows, _ = img.shape
    result = np.zeros(img.shape)

    # Step 1: Compute cumulative sum across each column ("Partial integral image")
    integral_image_cols = np.cumsum(img, axis= 0)

    # Step 2: Implement column-wise 1D box filter result with proper boundry handeling
    #        - neglet values the box would include outside of the image
    #        - use numpy.tile(...) to repeat an array/matrix

    # Method 1: (using np.tile() to repeat an array/matrix)

    # Image boundry (top)
    result[0: r+1, :] = np.tile(integral_image_cols[r,:],(r+1,1))
    # Regular areas (box fully overlapping with image)
    result[r + 1: rows - r, :] = integral_image_cols[2*r + 1:, :] - integral_image_cols[:rows - 2*r-1, :]
    # Image boundry (bottom)
    result[rows-r : rows, :] = np.tile(integral_image_cols[rows-1,:],(r,1))
 
    # #Method 2:
    # for row in range(rows):
    #    #Compute upper and lower bound for the box filter based on row index
    #    upper_bound = min(row + r, rows - 1)
    #    lower_bound = max(row-r-1, 0)
       
    #    # Calculate the box filter result for the current row by subtracting the cumulative sum of the lower bound from the upper bound
    #    result[row, :] = integral_image_cols[upper_bound, :] - integral_image_cols[lower_bound,:]
     
    return result

# Perform mean filtering for each row
def box_filter_rows(img, r):
    _, columns = img.shape
    result = np.zeros(img.shape)

    # Step 1: Compute cumulative sum along each row ('Partial integral Image')
    integral_image_rows = np.cumsum(img,axis=1)

    # Step 2: Impement the row-wise 1D box filter result with proper boundary handeling:
    # Image boundry (left)
    result[:, 0 : r+1] = np.tile(integral_image_rows[:,r][:,np.newaxis],(1,r+1))
    # Regular areas (box fully overlapping with image)
    result[:,r+1:columns-r] = (integral_image_rows[:,2*r+1:] - integral_image_rows[:, :columns-2*r-1])
    # Image boundry (right)
    result[:,columns-r:columns] = np.tile(integral_image_rows[:,columns-1][:,np.newaxis],(1,r))

    return result 


# Compute the (unnormalized) box- filtered image for a 2D input image and a given "radius" r (equal for both dimension)
def box_filter(img, r):
    rows, columns = img.shape

    # Check for valid input
    if rows < 2*r or columns < 2*r:
        raise ValueError('Error computing box filtering, value of r was selected too large')
    
    # Compute 2D filtered image by successively applying 1D filtering (once for each dimension)
    box_filtered_columns = box_filter_columns(img, r) # apply column wise
    result = box_filter_rows(box_filtered_columns,r) # apply row wise

    return result

# Compute the array of normalization constants for 2D box filter to create a mean filter
def get_box_norm(img, r):
    
    rows, columns = img.shape
    # result = np.zeros(img.shape)

    # for i in range(rows):
    #     for j in range(columns):
    #         # Calculate effective number of pixels in the box at the current position (i,j)
    #         num_pixels = (min(i+r+1, rows) - max(i-r,0)) * (min(j+r+1,columns) - max(j-r, 0))

    #         # Set the normalization constant in the result array
    #         if num_pixels == 0:
    #             result[i,j] = 0
    #         else:
    #             result[i,j] = 1 / num_pixels

    result =box_filter(np.ones([rows,columns]),r)

    return result


# Compute mean filtered image
def mean_filter(img, r, n=None):

    if n is None:
        result = box_filter(img,r) / get_box_norm(img,r)
    else:
        result = box_filter(img,r) / n

    return result

# Compute guided filter image from an input image g and guidance image i
def guided_filter(g, i, r, epsilon):

    # Normalization term
    n = get_box_norm(g,r)

    # (1a) Compute mean filtered images of the guidance and input image
    mean_i = mean_filter(i,r,n) 
    mean_g = mean_filter(g,r,n) 

    # (1b) Compute autocorrelation of guidance image and cross-correlation of input image
    corr_i = mean_filter(i*i, r,n)
    corr_ig = mean_filter(i*g, r,n)

    # (2) Calculate variance and covariance
    var_i = corr_i - mean_i * mean_i
    cov_ig = corr_ig - mean_i * mean_g

    # (3) Calculate a and b
    a = cov_ig / (var_i + epsilon)
    b = mean_g - a * mean_i

    # (4) Mean Filter a and b
    mean_a = mean_filter(a, r,n)
    mean_b = mean_filter(b,r,n)

    # (5) Compute the output image
    result = mean_a * i + mean_b

    return result  

def guided_filter_color(g, i, r, epsilon):

    rows = i[:,1,1].size
    columns = i[1,:,1].size

    n = box_filter(np.ones([rows,columns]),r)

    # Compute mean  of each color spaces
    mean_i_r = mean_filter(i[:,:,0],r,n)
    mean_i_g = mean_filter(i[:,:,1],r,n)
    mean_i_b = mean_filter(i[:,:,2],r,n)

    mean_g = mean_filter(g,r,n)

    mean_ig_r = mean_filter(i[:,:,0]*g, r, n)
    mean_ig_g = mean_filter(i[:,:,1]*g, r, n)
    mean_ig_b = mean_filter(i[:,:,2]*g, r, n)

    # Compute Covariance
    cov_ig_r = mean_ig_r - mean_i_r * mean_g
    cov_ig_g = mean_ig_g - mean_i_g * mean_g
    cov_ig_b = mean_ig_b - mean_i_b * mean_g

    # Compute Variance
    var_i_rr = mean_filter(i[:,:,0] * i[:,:,0],r,n) - mean_i_r * mean_i_r
    var_i_rg = mean_filter(i[:,:,0] * i[:,:,1],r,n) - mean_i_r * mean_i_g
    var_i_rb = mean_filter(i[:,:,0] * i[:,:,2],r,n) - mean_i_r * mean_i_b
    var_i_gg = mean_filter(i[:,:,1] * i[:,:,1],r,n) - mean_i_g * mean_i_g
    var_i_gb = mean_filter(i[:,:,1] * i[:,:,2],r,n) - mean_i_g * mean_i_b
    var_i_bb = mean_filter(i[:,:,2] * i[:,:,2],r,n) - mean_i_b * mean_i_b


    a = np.zeros([rows, columns, 3])

    for y in range(0, rows):
        for x in range(0, columns):
            variance_ = np.array([[var_i_rr[y,x], var_i_rg[y,x], var_i_rb[y,x]], \
                              [var_i_rg[y,x], var_i_gg[y,x], var_i_gb[y,x]], \
                              [var_i_rb[y,x], var_i_gb[y,x], var_i_bb[y,x]]])
            cov_ig = np.array([[cov_ig_r[y,x], cov_ig_g[y,x], cov_ig_b[y,x]]])

            a[y,x,:] = np.matmul(cov_ig, np.linalg.inv(variance_ + epsilon * np.identity(3)))

    b = mean_g - a[:,:,0]*mean_i_r - a[:,:,1]*mean_i_g - a[:,:,2]*mean_i_b

    q = mean_filter(a[:,:,0],r,n) * i[:,:,0] \
        + mean_filter(a[:,:,1],r,n) * i[:,:,1] \
        + mean_filter(a[:,:,2],r,n) * i[:,:,2] \
        + mean_filter(b,r,n)

    return q        



#################### Display Functions ######################################
def show_image(i,t,destroy_windows=True):
    cv2.imshow(t,i)

    print('Press any key to continue')
    cv2.waitKey(0)
    if destroy_windows:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()