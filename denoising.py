import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma, denoise_tv_chambolle, denoise_nl_means
import skimage.filters
from skimage.morphology import square


def median_filter(img_in, median_filter_size):

    '''
    Median filter denoising in each channel
    '''
    # Image out
    img_out = np.zeros(img_in.shape)

    # Split
    b,g,r = cv2.split(img_in)

    # Get median of each channel
    b = filter.median(b, skimage.morphology.square(width=median_filter_size), mode='reflect')
    g = filter.median(g, skimage.morphology.square(width=median_filter_size), mode='reflect')
    r = filter.median(r, skimage.morphology.square(width=median_filter_size), mode='reflect')

    # Add back to image
    img_out[:,:,0] = b
    img_out[:,:,1] = g
    img_out[:,:,2] = r

    return img_out



def denoise(image, method='BayesShrink', div_sigma=1, median_filter_size=3):

    # Estimate value of gaussian noise
    sigma_est = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
    print(f'- Estimated Gaussian noise standard deviation = {sigma_est}')
    
    if method=='BayesShrink':
        im_denoised = denoise_wavelet(image, channel_axis=-1, convert2ycbcr=True,
                                   method='BayesShrink', mode='soft',sigma=sigma_est/div_sigma,
                                   rescale_sigma=True)
    elif method == 'VisuShrink':
        im_denoised = denoise_wavelet(image, channel_axis=-1, convert2ycbcr=False,
                                        method='VisuShrink', mode='soft',
                                        sigma=sigma_est/div_sigma, rescale_sigma=True)

    elif method == 'MedianSquareFilter':
        im_denoised = median_filter(image, median_filter_size)

    return im_denoised, sigma_est


def post_denoise(image, initial_sigma, desired_ratio, scaling_const=100):
    '''
    '''
    window_size = get_best_window_size(image, initial_sigma, desired_ratio, scaling_const)
    print('- Applying post-denoising median filter with window size:',window_size)
    return denoise(image, method='MedianSquareFilter', median_filter_size=window_size)






def get_best_window_size(image, initial_sigma, desired_ratio, scaling_const):
    '''
    Works out how large the median filter needs to be to
    achieve a specific value of mean_brightness / sigma

    Taking median reduces error by roughly route N (~within 25%)
    Hence we pick a window size that reduce noise to desired
    amount for this image (noisier images will require 
    larger windows)

    Note: Make sure original image has not been scaled after
    estimating sigma!

    image: Initial image
    initial_sigma: Value of sigma before other denoising steps
    scaling_const: What factor has sigma been reduced by already?
                   (e.g. by previous denoising step). Set to 1
                   otherwise
    desired_ratio: What value of mean_pixel_value / sigma do we 
                   want to achieve? (e.g. >3? >10?)

    '''

    # Function for rounding to nearest odd number
    def round_to_nearest_odd(n):
        rounded = int(np.round((n + 1.00000001)/2)*2 - 1)
        if rounded < 0:
            rounded = 1
        return rounded

    # Number of vals required for this noise ratio
    n = ((desired_ratio*initial_sigma)/(np.mean(image)*scaling_const))**2

    # Sqrt to get the value needed for window size 
    dims = np.sqrt(n)

    # Round to odd number
    dims = round_to_nearest_odd(dims)

    return dims


