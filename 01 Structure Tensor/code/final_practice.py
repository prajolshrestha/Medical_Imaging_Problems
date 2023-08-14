import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


def main():
    # load an image
    image_rgb = cv2.imread('../P1_Structure_Tensor/data/rectangles.jpeg')
    show_image(image_rgb,'Original Image', destroy_windows=False)

    # Convert image to gray
    image_gray = convert2gray(image_rgb)
    show_image(image_gray,'Gray Image',destroy_windows=False)

    # Compute structure tensor
    J = compute_structure_tesnor(image_gray,sigma=0.5,rho=0.5,show=True)
    eigenvalues = compute_eigenvalues(J)
    c,e,f = generate_feature_masks(eigenvalues,thresh=0.003)
    show_feature_masks(c,e,f,True)

########### Functions for Calculation #####################

def convert2gray(image_rgb):
    temp = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0 # Normalize
    return image_gray

def filter_gauss(image, sigma):
    img_filtered = gaussian_filter(image,sigma,mode='constant')
    return img_filtered

def compute_gradient(image):
    # create two channels image for gradients ( initialize)
    img_gradient = np.empty((image.shape[0], image.shape[1],2))
    print(img_gradient.shape)

    # Convolve for forward differences in x direction
    x_kernel = np.asarray([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
    img_gradient[:,:,0] = convolve2d(image, x_kernel,mode='same')

    # Convolve for forward differences in y direction
    y_kernel = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=np.float32)
    img_gradient[:,:,1] = convolve2d(image,y_kernel,mode='same')

    return img_gradient

def compute_structure_tesnor(image_gray,sigma,rho,show):

    # Perform Gaussian Filtering
    image = filter_gauss(image_gray,sigma)
    show_image(image,' Gaussian blurred', destroy_windows=True)

    # Compute gradient tensor (f_x  & f_y)
    img_gradient = compute_gradient(image)
    if show:
        show_gradient(img_gradient)

    # Compute structure tensor (2 X 2)
    J = np.empty((image.shape[0],image.shape[1],2,2))
    f_x = img_gradient[:,:,0] 
    f_y = img_gradient[:,:,1]

    J[:,:,0,0] = f_x ** 2
    J[:,:,1,0] = f_x * f_x
    J[:,:,0,1] = f_x * f_x
    J[:,:,1,1] = f_y ** 2

    # Gaussian Filtering on tensor components
    for i in range(2):
        for j in range(2):
            J[:,:,i,j] = filter_gauss(J[:,:,i,j],rho)

    return J

def compute_eigenvalues(tensor):
    # each pixel ko str-tensor ko eigenvalue
    evs = np.empty((tensor.shape[0],tensor.shape[1],2))
    print(' Computing eigenvalues, this may take a while...')
    for y in range(tensor.shape[0]): # per row
        for x in range(tensor.shape[1]): # per cols
            t = tensor[y,x] # each pixel ko tensor
            ev,evec = np.linalg.eig(t)
            evs[y,x] = ev

    return evs

def generate_feature_masks(evs, thresh = 0.003):

    corners = np.zeros(np.shape(evs[:,:,0]))
    straight_edges = np.zeros(np.shape(evs[:,:,0]))
    flat_areas = np.zeros(np.shape(evs[:,:,0]))

    for i in range(0,evs.shape[0]): # per row
        for j in range(0, evs.shape[1]): # per cols

            eigenvalues = evs[i,j] # eigenvalue at pixel(i,j)
            # Cornerness measure
            corner_responses = eigenvalues[0]*eigenvalues[1] - 0.04*(eigenvalues[0] + eigenvalues[1])**2

            # Classify based on corner response
            if corner_responses > thresh:
                corners[i,j] = 1.0
            elif corner_responses < -thresh:
                straight_edges[i,j] = 1.0
            else:
                flat_areas[i,j] = 1.0

    return corners, straight_edges, flat_areas
            


    
################# Functions for displaying diff. Images #######


def show_image(i,t,destroy_windows = True):
    cv2.imshow(t,i)
    
    print('Press a key to continue...')
    cv2.waitKey(0)
    if destroy_windows:
        cv2.destroyAllWindows()

def show_gradient(img_gradient, destroy_windows =True):
    # Rescale image for display purpose
    show_image(img_gradient[:,:,0]/ 2.0 + 0.5, 'x gradients', destroy_windows=False)
    show_image(img_gradient[:,:,1]/ 2.0 + 0.5, 'y gradients', destroy_windows=False)
    
    # compute norm = sqrt(x^2 + y^2)
    img_gradient_norm = np.sqrt(img_gradient[:,:,0] * img_gradient[:,:,0] + img_gradient[:,:,1] * img_gradient[:,:,1])
    show_image(img_gradient_norm,'gradient L2-norm', destroy_windows)
    

def show_feature_masks(c,e,f,destroy_windows = True):
    print('|Features are indicated by white.|')
    show_image(c, 'Corners', False)
    show_image(e, 'Straight Edges', False)
    show_image(f, 'Flat Areas', False)



if __name__ == '__main__':
    main()