import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma, denoise_tv_chambolle, denoise_nl_means


def denoise(im, method='BayesShrink', div_sigma=1, nl_means_fast=True, tv_chambolle_iter=10):
    
    # Estimate noise
    sigma_est = estimate_sigma(im, channel_axis=-1, average_sigmas=True)
#     print(f'Estimated Gaussian noise standard deviation = {sigma_est}')
    
    if method=='wavelet_BayesShrink':
        im_denoised = denoise_wavelet(im, channel_axis=-1, convert2ycbcr=True,
                                   method='BayesShrink', mode='soft',
                                   rescale_sigma=True)
    elif method == 'wavelet_VisuShrink':
        im_denoised = denoise_wavelet(im, channel_axis=-1, convert2ycbcr=True,
                                        method='VisuShrink', mode='soft',
                                        sigma=sigma_est/div_sigma, rescale_sigma=True)
    elif method == 'tv_chambolle': # Very slow!!!
        im_denoised = denoise_tv_chambolle(im, weight=0.1, eps=0.0002, max_num_iter=tv_chambolle_iter)
    
    elif method == 'nl_means':
        patch_kw = dict(patch_size=2,      # 5x5 patches
                patch_distance=2,  # 13x13 search area
                channel_axis=-1)
        im_denoised = denoise_nl_means(im, h=sigma_est/div_sigma, fast_mode=nl_means_fast, **patch_kw)

    return im_denoised