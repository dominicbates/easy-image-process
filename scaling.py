import pandas as pd
import numpy as np
import cv2
from skimage.exposure import match_histograms, histogram


def histogram_equalization(img_in, max_val = 256, bits='uint8'):
    
    max_val = 256
    bits = 'uint8'
    # segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), max_val, [0, max_val])
    h_g, bin_g = np.histogram(g.flatten(), max_val, [0, max_val])
    h_r, bin_r = np.histogram(r.flatten(), max_val, [0, max_val])
    
    # calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    

    # mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*(max_val-1)/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype(bits)
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*(max_val-1)/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype(bits)
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*(max_val-1)/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype(bits)

    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
    img_out = cv2.merge((img_b, img_g, img_r))

    return img_out



def histogram_equalization_16(img_in):
    
    # For storing image
    img_out = np.zeros(img_in.shape)
    
    # segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 65536, [0, 65536])
    h_g, bin_g = np.histogram(g.flatten(), 65536, [0, 65536])
    h_r, bin_r = np.histogram(r.flatten(), 65536, [0, 65536])
    
    # calculate cdf    
    cdf_b = h_b.cumsum()
    cdf_g = h_g.cumsum()
    cdf_r = h_r.cumsum()
    
    # mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*65535/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint16')
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*65535/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint16')
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*65535/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint16')

    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
    
    # Add to image
    img_out[:,:,0] = img_b
    img_out[:,:,1] = img_g
    img_out[:,:,2] = img_r

    return img_out




def get_average_hist(images):
    
    '''
    Calculates the average histogram from a set of images
    (mostly just for use in below functions)
    '''
    import copy
    from skimage.exposure import histogram

    # Blank arrays for storing values
    histogram_counts_r = np.zeros(256)
    histogram_counts_g = np.zeros(256)
    histogram_counts_b = np.zeros(256)
    histogram_bins = np.arange(256)

    # Loop through all images
    for image in images:
        # N. pixels
        n_pixels = image.shape[0]*image.shape[1]
        
        # Get hist of this image
        image_hist = histogram(image, nbins=256, channel_axis=-1)
        bins = image_hist[1]

        # Add values to correct size array (256)
        hr = np.zeros(256)
        hg = np.zeros(256)
        hb = np.zeros(256)
        hr[bins[0]:(bins[-1]+1)] = image_hist[0][0]/n_pixels
        hg[bins[0]:(bins[-1]+1)] = image_hist[0][1]/n_pixels
        hb[bins[0]:(bins[-1]+1)] = image_hist[0][2]/n_pixels

        # Add cumilative counts per channel
        histogram_counts_r += hr
        histogram_counts_g += hg
        histogram_counts_b += hb

    # Normalize
    histogram_counts_r = histogram_counts_r/len(images)
    histogram_counts_g = histogram_counts_g/len(images)
    histogram_counts_b = histogram_counts_b/len(images)
    counts = [histogram_counts_r, histogram_counts_g, histogram_counts_b]
    
    return counts, histogram_bins

    

def get_average_hist_16(images):
    
    '''
    Calculates the average histogram from a set of images
    (mostly just for use in below functions)
    '''
    import copy
    from skimage.exposure import histogram

    # Blank arrays for storing values
    histogram_counts_r = np.zeros(65536)
    histogram_counts_g = np.zeros(65536)
    histogram_counts_b = np.zeros(65536)
    histogram_bins = np.arange(65536)

    # Loop through all images
    for image in images:
        # N. pixels
        n_pixels = image.shape[0]*image.shape[1]
        
        # Get hist of this image
        image_hist = histogram(image, nbins=65536, channel_axis=-1, source_range='dtype')
        bins = image_hist[1]

        # Add values to correct size array (256)
        hr = np.zeros(65536)
        hg = np.zeros(65536)
        hb = np.zeros(65536)
        hr[bins[0]:(bins[-1]+1)] = image_hist[0][0]/n_pixels
        hg[bins[0]:(bins[-1]+1)] = image_hist[0][1]/n_pixels
        hb[bins[0]:(bins[-1]+1)] = image_hist[0][2]/n_pixels

        # Add cumilative counts per channel
        histogram_counts_r += hr
        histogram_counts_g += hg
        histogram_counts_b += hb

    # Normalize
    histogram_counts_r = histogram_counts_r/len(images)
    histogram_counts_g = histogram_counts_g/len(images)
    histogram_counts_b = histogram_counts_b/len(images)
    counts = [histogram_counts_r, histogram_counts_g, histogram_counts_b]
    
    return counts, histogram_bins


def create_reference_image(images, n_pixels=1000):
    '''
    Creates a "reference image" from a set of images
    The reference image is just random noise, but 
    with the same histogram as the average of all 
    the reference images (i.e. can be used for histogram
    matching to the reference)
    '''
    # get average histogram
    counts, histogram_bins = get_average_hist(images)

    # Create randomly sampled image with these statistics
    reference_size = n_pixels
    r_channel = np.random.choice(histogram_bins, [reference_size,reference_size], p=counts[0])
    g_channel = np.random.choice(histogram_bins, [reference_size,reference_size], p=counts[1])
    b_channel = np.random.choice(histogram_bins, [reference_size,reference_size], p=counts[2])
    reference_image = cv2.merge([r_channel, g_channel, b_channel])
    return reference_image


def create_reference_image_16(images, n_pixels=1000):
    '''
    Creates a "reference image" from a set of images
    The reference image is just random noise, but 
    with the same histogram as the average of all 
    the reference images (i.e. can be used for histogram
    matching to the reference)
    '''
    # get average histogram
    counts, histogram_bins = get_average_hist_16(images)

    # Create randomly sampled image with these statistics
    reference_size = n_pixels
    r_channel = np.random.choice(histogram_bins, [reference_size,reference_size], p=counts[0])
    g_channel = np.random.choice(histogram_bins, [reference_size,reference_size], p=counts[1])
    b_channel = np.random.choice(histogram_bins, [reference_size,reference_size], p=counts[2])
    reference_image = cv2.merge([r_channel, g_channel, b_channel])
    return reference_image



def histogram_matching(img_in, reference_images):
    '''
    Matches the histogram of an image to those of a 
    set of reference images
    '''
    reference_image = create_reference_image(images)
    return match_histograms(img_in, reference_image, channel_axis=-1)
    
    
def match_reference_image(img_in, reference_image):
    '''
    Matches the histogram of an image to a 
    reference images (can be created using 
    create_reference_image())
    '''
    return match_histograms(img_in, reference_image, channel_axis=-1)



def stretch_to_format(image, image_type='uint16'):
    '''
    Stretches image so min and max values are 0 and 255 (or 65535 for 16 bit)
    '''

    # Check input
    if image_type not in ['uint16', 'uint8']:
        raise ValueError('"image_type" must be either "uint16" or "uint8"') 
    
    # Scale image so min = 0
    image = np.array(image, dtype='float')
    min_val = image.min()
    image = image - min_val
    max_val = image.max()

    # Transform to right type:
    if image_type == 'uint16':
        return np.array((65535*image/max_val).astype(int),dtype ='uint16')
    elif image_type == 'uint8':
        return np.array((255*image/max_val).astype(int),dtype ='uint8')




