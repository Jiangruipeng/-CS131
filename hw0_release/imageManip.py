import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5* np.sqrt(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2grey(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    image_height = image.shape[0]
    image_width = image.shape[1]
    r = image[:,:,0].reshape(image_height,image_width,1)
    g = image[:,:,1].reshape(image_height,image_width,1)
    b = image[:,:,2].reshape(image_height,image_width,1)
    

    zero_data = np.zeros(image_height*image_width).reshape(image_height,image_width,1)

    #out = image.copy()
    if(channel == "R"):
        out = np.concatenate((zero_data,g,b),axis=2).reshape(image_height,image_width,3)
    elif(channel == "G"):
        out = np.concatenate((r,zero_data,b),axis=2).reshape(image_height,image_width,3)
    elif(channel == "B"):
        out = np.concatenate((r,g,zero_data),axis=2).reshape(image_height,image_width,3)
    ### END YOUR CODE
    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    image_height = image.shape[0]
    image_width = image.shape[1]
    zero_data = np.zeros(image_height*image_width).reshape(image_height,image_width,1)


    l = lab[:,:,0].reshape(image_height,image_width,1)
    a = lab[:,:,1].reshape(image_height,image_width,1)
    b = lab[:,:,2].reshape(image_height,image_width,1)
    
    out = None

    ### YOUR CODE HERE
    if(channel == "L"):
        out = np.concatenate((l,zero_data,zero_data),axis=2).reshape(image_height,image_width,3)
        #out=color.lab2rgb(lab)
        out = color.lab2rgb(out)
    elif(channel == "A"):
        out = np.concatenate((zero_data,a,zero_data),axis=2).reshape(image_height,image_width,3)
        out = color.lab2rgb(out)
    elif(channel == "B"):
        out = np.concatenate((zero_data,zero_data,b),axis=2).reshape(image_height,image_width,3)
        out = color.lab2rgb(out)



    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    image_height = image.shape[0]
    image_width = image.shape[1]
    zero_data = np.zeros(image_height*image_width).reshape(image_height,image_width,1)


    h = hsv[:,:,0].reshape(image_height,image_width,1)
    s = hsv[:,:,1].reshape(image_height,image_width,1)
    v = hsv[:,:,2].reshape(image_height,image_width,1)
    
    out = None

    ### YOUR CODE HERE
    if(channel == "H"):
        out = np.concatenate((h,zero_data,zero_data),axis=2).reshape(image_height,image_width,3)
        #out=color.lab2rgb(lab)
        out = color.hsv2rgb(out)
    elif(channel == "S"):
        out = np.concatenate((zero_data,s,zero_data),axis=2).reshape(image_height,image_width,3)
        out = color.hsv2rgb(out)
    elif(channel == "V"):
        out = np.concatenate((zero_data,zero_data,v),axis=2).reshape(image_height,image_width,3)
        out = color.hsv2rgb(out)

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    image_width = image1.shape[1]

    
    part1 = rgb_exclusion(image1, channel1)
    part2 = rgb_exclusion(image2, channel2)

    part1=part1[:, 0:int(image_width/2), :]
    part2=part2[:, int(image_width/2):, :]
    out = np.concatenate((part1,part2),axis=1)

    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
