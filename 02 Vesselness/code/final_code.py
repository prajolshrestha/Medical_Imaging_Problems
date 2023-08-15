import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def main():

    # initialization of constants
    beta = 0.5
    c = 0.08

    # load and prepare image
    image_rgb = cv2.imread('../P2_Vesselness/data/coronaries.jpg')
    show_image(image_rgb,'Original image', destroy_windows=True)
    print(image_rgb.shape)
    image_gray = convert2gray(image_rgb)
    show_image(image_gray,'gray image', destroy_windows=True)

    # Compute Vesselness
    scales = [1.0, 1.5, 2.0, 3.0]
    images_vesselness = []
    for s in scales:
        images_vesselness.append(calculate_vesselness_2d(image_gray,s,beta,c))

    # Compute scale maximum
    result = compute_scale_maximum(images_vesselness)
    show_four_scales(image_gray, result, images_vesselness, scales)

    

   
################ Functions for Calculations ##############

def convert2gray(image_rgb):
    temp = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0
    return image_gray


def compute_hessian(image, sigma):
    # Gaussian filtering
    image_gauss = gaussian_filter(image,sigma,mode='constant')

    print('Computing Hessian ...')
    # First method ##############################################
    # First order Gradient
    x_kernel = np.asarray([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=np.float32)
    f_x = convolve2d(image_gauss,x_kernel,mode='same')
    y_kernel = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=np.float32)
    f_y = convolve2d(image_gauss,y_kernel,mode='same')
    
    # # create two channels image for gradients ( initialize)
    # img_gradient = np.empty((image.shape[0], image.shape[1],2))
    # img_gradient[:,:,0] =f_x
    # img_gradient[:,:,1] =f_y
    # show_gradient(img_gradient,destroy_windows=True)
    
    # Second order Gradient
    fxx = convolve2d(f_x,x_kernel,mode='same')
    fxy = convolve2d(f_x,y_kernel,mode='same')
    fyx = convolve2d(f_y,x_kernel,mode='same')
    fyy = convolve2d(f_y,y_kernel,mode='same')
    
    # Second Method ############################################
    # Laplacian of Gaussian (LoG) filter
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # kernel_xy = np.array([[1, 0, 1], [0, 0, 0], [-1, 0, -1]])
    # fxx = convolve2d(image_gauss, kernel, mode='same')
    # fyy = convolve2d(image_gauss, kernel.T, mode='same')  # Transpose for second derivative along y
    # fxy = fyx = convolve2d(image_gauss, kernel_xy, mode='same')

    # Scale Normalization
    fxx = fxx * sigma**2
    fxy = fxy * sigma**2 
    fyx = fyx * sigma**2 
    fyy = fyy * sigma**2 

    # Save values in a single array
    H = np.zeros((image_gauss.shape[0], image_gauss.shape[1],2,2))
    H[:,:,0,0] = fxx
    H[:,:,0,1] = fxy
    H[:,:,1,0] = fyx
    H[:,:,1,1] = fyy

    print('...done.')
    return H


def calculate_vesselness_2d(image,scale,beta,c):
    vesselness = np.zeros(image.shape)

    # Compute Hessian for each pixel
    H = compute_hessian(image, scale)

    # Compute Eigenvalues for the Hessians
    eigenvalues = compute_eigenvalues(H)

    print('Computing vesselness ...')
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            lambda1, lambda2 = eigenvalues[y,x]
            v = vesselness_measure(lambda1, lambda2, beta, c)
            vesselness[y,x] = v

    print('..done.')
    return vesselness


def compute_eigenvalues(hessian):

    evs = np.empty((hessian.shape[0],hessian.shape[1],2))
    print('Computing eigenvalues, this may take a while...')

    for y in range(hessian.shape[0]):
        for x in range(hessian.shape[1]):
            temp = hessian[y,x]
            ev,_ = np.linalg.eig(temp)
            evs[y,x] = ev
    print('...done.')
    return evs

def vesselness_measure(lambda1, lambda2, beta, c):

    # ensure lambda1 >= lambda2
    lambda1, lambda2 = sort_descending(lambda1,lambda2)
    
    # the vesselness measure is zero if lambda1 is positive (inverted/dark vessel)
    # if both eigenvalues are zero, set RB and S to zero, otherwise compute them as shown in the course
   
    if lambda1 > 0:
        v = 0.0
        return v
    
    if lambda1 == 0 and lambda2 == 0:
        Rb = 0.0
        S = 0.0   
    else:
        Rb = lambda2 / lambda1
        S = np.sqrt(lambda1**2 + lambda2**2)
        
    v = np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * c **2)))
    return v



def sort_descending(value1, value2):

    if np.abs(value1) < np.abs(value2):
        buf = value2
        value2 = value1
        value1 = buf
    
    return value1, value2

def compute_scale_maximum(image_list):
    result = image_list[0]
    print('Computing maximum...')

    # Compute pixel-wise maximum from all images in image_list
    for img in image_list[1:]:
        result = np.maximum(result,img)

    print('...done.')
    return result

############### Functions for Dispaying images ##########
def  show_image(i,t, destroy_windows = True):
    
    cv2.imshow(t,i)

    print('Press any key to continue.')
    cv2.waitKey(0)
    if destroy_windows:
        cv2.destroyAllWindows()

def prepare_subplot_image(image, title='', idx = 1):
    if idx > 6:
        return
    
    plt.gcf()
    plt.subplot(2,3,idx)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image,cmap='gray', vmin=0, vmax= np.max(image))


def show_four_scales(original, result, image_list, scales):

    plt.figure('Vesselness')

    prepare_subplot_image(original, 'original', 1)
    prepare_subplot_image(image_list[0], 'sigma = ' +str(scales[0]),2)
    prepare_subplot_image(image_list[1], 'sigma = ' +str(scales[1]),3)
    prepare_subplot_image(result, 'result', 4)
    prepare_subplot_image(image_list[2], 'sigma = ' +str(scales[2]),5)
    prepare_subplot_image(image_list[3], 'sigma = ' +str(scales[3]),6)
    
    plt.show()

def show_gradient(img_gradient, destroy_windows =True):
    # Rescale image for display purpose
    show_image(img_gradient[:,:,0]/ 2.0 + 0.5, 'x gradients', destroy_windows=False)
    show_image(img_gradient[:,:,1]/ 2.0 + 0.5, 'y gradients', destroy_windows=False)

if __name__ == '__main__':
    main()