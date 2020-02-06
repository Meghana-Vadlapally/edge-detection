#!/usr/bin/env python
# coding: utf-8

# In[21]:


import copy
import os

import cv2
import numpy as np


# In[22]:


# Prewitt operator
prewitt_x = [[1, 0, -1]] * 3
prewitt_x = np.asarray(prewitt_x)


prewitt_y = [[1] * 3, [0] * 3, [-1] * 3]
prewitt_y = np.asarray(prewitt_y)

# Sobel operator
sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_x=np.asarray(sobel_x)


sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
sobel_y=np.asarray(sobel_y)


# In[23]:


def flip_x(img):
    """Flips a given image along x axis."""
    flipped_img = copy.deepcopy(img)
    center = int(len(img) / 2)
    for i in range(center):
        flipped_img[i] = img[(len(img) - 1) - i]
        flipped_img[(len(img) - 1) - i] = img[i]
    return flipped_img

def flip_y(img):
    """Flips a given image along y axis."""
    flipped_img = copy.deepcopy(img)
    center = int(len(img[0]) / 2)
    for i, row in enumerate(img):
        for j in range(center):
            flipped_img[i][j] = img[i][(len(img[0]) - 1) - j]
            flipped_img[i][(len(img[0]) - 1) - j] = img[i][j]
    return flipped_img

def flip2d(img, axis=None):
    """Flips an image along a given axis.

    Hints:
        Use the function flip_x and flip_y.

    Args:
        img: nested list (int), the image to be flipped.
        axis (int or None): the axis along which img is flipped.
            if axix is None, img is flipped both along x axis and y axis.

    Returns:
        flipped_img: nested list (int), the flipped image.
    """
    # TODO: implement this function.
    #raise NotImplementedError
    if axis == 'x':
        
        flipped_img = flip_x(img)
        
    elif axis == 'y':
        
        flipped_img = flip_y(img)
        
    else:
        
        print('Please enter valid axis')
    
    return flipped_img


# In[25]:


def read_image(img_path, show = True):
    
    """
        Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img) 
        
    img = [list(row) for row in img]
    print(np.shape(img))
    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    #print(img.shape)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
 



def write_image(img, img_saving_path):
    
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    
    """
              Writes an image to a given path.             
    
    elif isinstance(img, np.ndarray):       

        if not img.dtype == np.uint8:
            assert np.max(img) <= 1.3, "Maximum pixel value {:.3f} is greater than 1.3".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")
        
    """

    cv2.imwrite(img_saving_path, img)     
       





    
    
def convolve2d(imageArray, kernel):
    """Convolves a given image and a given kernel.

    Steps:
        (1) flips the either the img or the kernel.
        (2) pads the img or the flipped img.
            this step handles pixels along the border of the img,
            and makes sure that the output img is of the same size as the input image.
        (3) applies the flipped kernel to the image or the kernel to the flipped image,
            using nested for loop.

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel.

    Returns:
        img_conv: nested list (int), image.
    """
    # TODO: implement this function.
    
    
    #imageArray = flip2d( imageArray, axis = 'y')
    #imageArray = flip2d( imageArray, axis = 'x')
    

    imageRows  = imageArray.shape[0]
    imageColumns = imageArray.shape[1]
    
    print(imageRows)
    print(imageColumns)
    
    kernelRows = kernel.shape[0]
    kernelColumns = kernel.shape[1]
    
    print(kernelRows)
    print(kernelColumns)
    
    rows = kernelRows // 2
    columns = kernelColumns // 2
    
    img_conv = np.zeros(shape=(imageRows, imageColumns))
    
    print('empty array of convoluted image', img_conv, img_conv.shape)
    
    for i in range(rows, imageRows-rows):
        for j in range(columns, imageColumns-columns):
            sum = 0
            for k in range(kernelRows):
                for l in range(kernelColumns):
                    #sum = elementwise_add( sum, elementwise_mul(kernel[k][l], imageArray[i-rows+k][j-columns+l]) )
                    sum = sum + kernel[k][l] * imageArray[i-rows+k][j-columns+l]
            img_conv[i][j] = sum
    print('convoluted image shape', img_conv.shape)
    
    
    
    
                
    return img_conv

       #raise NotImplementedError
        


def normalize(imageArray):
    """Normalizes a given image.

    Hints:
        Noralize a given image using the following equation:

        normalized_img = frac{img - min(img)}{max(img) - min(img)},

        so that the maximum pixel value is 255 and the minimum pixel value is 0.

    Args:
        img: nested list (int), image.

    Returns:
        normalized_img: nested list (int), normalized image.
    """
    # TODO: implement this function.
    
    #converting img list to array
    
    

    imageRows  = imageArray.shape[0]
    imageColumns = imageArray.shape[1]

    normalized_img = np.zeros(shape=(imageRows,imageColumns))
    
    print('normalized array before normalization', normalized_img, normalized_img.shape)
    
    mini = np.min(imageArray)
    
    maxi = np.max(imageArray)
    
    for i in range(imageRows):
        for j in range(imageColumns):
            normalized_img[i][j] = ( (imageArray[i][j] - mini) / (maxi - mini) ) 
            
            #print('element', normalized_img[i][j])
            
            

            if normalized_img[i][j] > 0.5:
                normalized_img[i][j] = 0
            else:
                normalized_img[i][j] = 255
                
            
                
     
                    

    print('after normalization', normalized_img, normalized_img.shape )
    #raise NotImplementedError
    return normalized_img


def detect_edges(imageArray, kernel, norm=True):
    """Detects edges using a given kernel.

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel used to detect edges.
        norm (bool): whether to normalize the image or not.

    Returns:
        img_edge: nested list (int), image containing detected edges.
    """
    # TODO: detect edges using convolve2d and normalize the image containing detected edges using normalize.
    #raise NotImplementedError

    img_conv = convolve2d(imageArray, kernel)  
    
    img_edges = normalize(img_conv)

    return img_edges

def edge_magnitude(edge_x, edge_y):
    """Calculate magnitude of edges by combining edges along two orthogonal directions.

    Hints:
        Combine edges along two orthogonal directions using the following equation:

        edge_mag = sqrt(edge_x ** 2 + edge_y **).

        Make sure that you normalize the edge_mag, so that the maximum pixel value is 1.

    Args:
        edge_x: nested list (int), image containing detected edges along one direction.
        edge_y: nested list (int), image containing detected edges along another direction.

    Returns:
        edge_mag: nested list (int), image containing magnitude of detected edges.
    """
    # TODO: implement this function.
    
    print('edge x shape', edge_x.shape)
    
    print('edge Y shape', edge_y.shape)
    
    edgeXR= edge_x.shape[0]
    edgeXC= edge_x.shape[1]
    
    edgeYR= edge_y.shape[0]
    edgeYC= edge_y.shape[1]
    
    
    
    if edgeXR == edgeYR and edgeXC == edgeYC :
        rows = edgeXR
        columns = edgeYC
    else:
        print('OOPS!! something is wrong!')
    
    print('edge mag rows', rows)
    print('edge mag columns', columns)
    
    edge_mag = np.zeros(shape=(rows,columns))
    
    print('edge magnitude array before: ', edge_mag, edge_mag.shape)
        
    for i in range(rows):        
        for j in range(columns):
            edge_mag[i][j] = ((edge_x[i][j]**2 + edge_y[i][j]**2) ** (1/2) )
            
    print('edge mag shape and array after: ', edge_mag, edge_mag.shape)
    
    
    
    #raise NotImplementedError
    return edge_mag


def main():   
    
    
    imagePath = input('enter the image path ')

    kernelSelection = input('select kernel, you can choose from Prewitt or prewitt; sobel or Sobel ')

    resultDirectory = input('please enter the directory to save the result ')   



    img = read_image(imagePath)

    # changing image list to array

    imageArray = np.asarray(img)

    print(imageArray.shape)

    
    imageArray = cv2.medianBlur(imageArray, 25)
    imageArray = cv2.GaussianBlur(imageArray,(3,3),cv2.BORDER_DEFAULT)
    #imageArray = cv2.medianBlur(imageArray, 55)
    
  
    

    if kernelSelection in ["prewitt", "Prewitt"]:
        kernel_x = prewitt_x
        kernel_y = prewitt_y
        print(kernel_x,kernel_y)
    elif kernelSelection in ["sobel", "Sobel"]:
        kernel_x = sobel_x
        kernel_y = sobel_y
        print('kernel ',kernel_x)
        print('kernel shape', kernel_x.shape)
    else:
        raise ValueError("Kernel type not recognized.")

    if not os.path.exists(resultDirectory):
        os.makedirs(resultDirectory)
 
    img_edge_x = detect_edges(imageArray, kernel_x, False)
    #print(img_edge_x.shape)
    write_image(normalize(img_edge_x), os.path.join(resultDirectory, "{}_edge_x.jpeg".format(kernelSelection.lower())))

    img_edge_y = detect_edges(imageArray, kernel_y, False)
    write_image(normalize(img_edge_y), os.path.join(resultDirectory, "{}_edge_y.jpeg".format(kernelSelection.lower())))
    
    img_edges = edge_magnitude(img_edge_x, img_edge_y)
    write_image(img_edges, os.path.join(resultDirectory, "{}_edge_mag.jpg".format(kernelSelection.lower())))
    
     
    
           
    
    
    #theta = np.arctan2(img_edge_y, img_edge_x)
    
    
    #imageFinal = non_max_suppression(normalize(img_edges), theta) 
    
    #write_image(imageFinal, os.path.join(resultDirectory, "{}_edge_final.jpeg".format(kernelSelection.lower()))) 
    

    
    
    



if __name__ == "__main__":
    main()
    
    
    
                
                
                
    


# In[ ]:




