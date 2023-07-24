# easy-image-process

The repository contains a few functions for auto-scaling and denoising images (in `scaling.py` and `denoising.py`). The general usage will be to apply one of the auto-scaling methods, and then apply one of the denoising methods on top of this. For particularly noisy images, you might want to apply a second denoising step.

# Installation

No proper requirements, but requires installation of:
- `skimage`
- `cv2` (opencv-python)

`imageio` can also be used for each loading of images, although is not requred anywhere in the code

# scaling.py

`scaling.py` contains a few different methods of auto scaling images:

- For "equalising" the histograms of an image you can run: `scaled_image = histogram_equalization(image)`. This has uint8 and uint16 options
- For transforming an image to match the histograms of a reference image you can run: `scaled_image = match_reference_image(image, reference_image)` 
- If you want to match the statistics of a large set of reference images (rather than an individual image) you can run the following:
	```
	reference_image = create_reference_image([image1, image2, ...])
	scaled_image = match_reference_image(image, reference_image)
	```
	This creates a single image of random noise drawn from the the statistics of the full sample, and then uses this to match to. There are also uint8 and uint16 options here
- The code also contains some functions for transforming/scaling images to uint8 and uint16 format


# denoising.py

`denoising.py` containg a single function for denoising images, with multiple options for method. The most succesful denoising method seems to be wavelet denoising with "BayesShrink").

Example usage would be: `denoised_image = denoise(image, method='BayesShrink')`

# Example Code

Full process is very easy. Just run a scaling step, followed by a denoising step if desired. 
```
# Import functions
from scaling import histogram_equalization_16
from denoising import denoise, post_denoise

# Scale image using equalisation
scaled_image = histogram_equalization_16(image)

# Denoise this image using wavelet & BayesShrink
denoised_image, sigma_est = denoise(scaled_image, method='BayesShrink', div_sigma=0.25) # 0.25 means more denoising

# Post-denoising step to clean up some more if needed
full_cleaned_image = post_denoise(denoised_image, sigma_est, desired_ratio=3, scaling_const=100)

# Re-scaling if desired
# ...
```


