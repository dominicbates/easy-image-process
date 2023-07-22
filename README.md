# easy-image-process

The repository contains a few simple functions for auto-scaling and denoising images (in `scaling.py` and `denoising.py`). The general usage will be to apply one of the two auto-scaling methods, and then apply one of the denoising methods on top of this.

## Installation

No proper requirements, but requires installation of:
- `skimage`
- `cv2` (opencv-python)

`imageio` can also be used for each loading of images, although is not requred anywhere in the code

## scaling.py

`scaling.py` contains two different methods of auto scaling images:

- `scaled_image = histogram_equalization(image)` which can be used to equalise the histograms of an image
- `scaled_image = match_reference_image(image, reference_image)` which can be used to match transform an image to match the histograms of a reference image. 
- For matching the statistics of a large set of reference images:
	`reference_image = match_reference_image([image1, image2, ...])`
	`scaled_image = match_reference_image(image, reference_image)`


## denoising.py

`denoising.py` containg a single function for denoising images, with multiple options for method. The most succesful denoising method seems to be wavelet denoising (for which you can use wither "BayesShrink" or "VisuShrink"

Example usage would be: `denoised_image = denoise(image, method='BayesShrink')`

## Example Code

Full scaling process is very easy. Just run a scaling step, followed by a denoising step, e.g.:
`scaled_image = histogram_equalization(image)
denoised_image = denoise(scaled_image, method='BayesShrink')`



