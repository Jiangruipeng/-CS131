"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height=int((Hk-1)/2)
    pad_width=int((Wk-1)/2)
    image_pad = zero_pad(image,pad_height,pad_width)
    for hi in range(Hi):
        for wi in range(Wi):
            val_s = 0
            for hk in range(Hk):
                for wk in range(Wk):
                    idhx = hk-pad_height#kernel
                    idhy = wk-pad_width#kernel
                    idfx = hi - idhx#image
                    idfy = wi - idhy#image
                    val_s = val_s+ kernel[hk,wk]*image_pad[idfx+pad_height,idfy+pad_width]
            out[hi, wi]= val_s
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    H_new = int(H+2*pad_height)
    W_new = int(W+2*pad_width)
    out = np.zeros((H_new, W_new))
    for h in range(H_new):
        if((h-pad_height>=0) & (h-pad_height <H)):
            for w in range(W_new):
                if((w -pad_width>=0) & (w -pad_width< W)):
                    out[h,w]=image[h-pad_height, w-pad_width]
                    
            
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height=int(np.ceil((Hk-1)/2))
    pad_width=int(np.ceil((Wk-1)/2))
    image_pad = zero_pad(image,pad_height,pad_width)
    
    arr_h = np.flip(np.flip(kernel, axis=0),axis=1)    
    for hi in range(Hi):
        for(wi) in range(Wi):
            arr_f = image_pad[hi:hi+Hk, wi:wi+Wk]
            out[hi,wi] = np.sum(arr_f*arr_h)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    image=f
    kernel=g
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height=int(np.ceil((Hk-1)/2))
    pad_width=int(np.ceil((Wk-1)/2))
    image_pad = zero_pad(image,pad_height,pad_width)
    
    arr_h = kernel    
    for hi in range(Hi):
        for(wi) in range(Wi):
            arr_f = image_pad[hi:hi+Hk, wi:wi+Wk]
            out[hi,wi] = np.sum(arr_f*arr_h)/(np.sqrt(np.sum(arr_f*arr_f))*np.sqrt(np.sum(arr_h*arr_h)))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    image=f
    kernel=g
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height=int(np.ceil((Hk-1)/2))
    pad_width=int(np.ceil((Wk-1)/2))
    image_pad = zero_pad(image,pad_height,pad_width)
    
    arr_h = kernel - np.mean(kernel)   
    for hi in range(Hi):
        for(wi) in range(Wi):
            arr_f = image_pad[hi:hi+Hk, wi:wi+Wk]
            arr_f = arr_f - np.mean(arr_f)
            out[hi,wi] = np.sum(arr_f*arr_h)/(np.sqrt(np.sum(arr_f*arr_f))*np.sqrt(np.sum(arr_h*arr_h)))

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    image=f
    kernel=g
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height=int(np.ceil((Hk-1)/2))
    pad_width=int(np.ceil((Wk-1)/2))
    image_pad = zero_pad(image,pad_height,pad_width)
    
    arr_h = (kernel - np.mean(kernel))/np.std(kernel)  
    for hi in range(Hi):
        for(wi) in range(Wi):
            arr_f = image_pad[hi:hi+Hk, wi:wi+Wk]
            arr_f = (arr_f - np.mean(arr_f))/np.std(arr_f)
            out[hi,wi] = np.sum(arr_f*arr_h)/(np.sqrt(np.sum(arr_f*arr_f))*np.sqrt(np.sum(arr_h*arr_h)))

    return out
