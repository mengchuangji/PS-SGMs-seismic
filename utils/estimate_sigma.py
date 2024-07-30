import skimage.data
from skimage.restoration import estimate_sigma
from skimage import img_as_float
import numpy as np
# img = img_as_float(skimage.data.camera())
# sigma = 0.1
# rng = np.random.default_rng()
# img = img + sigma * rng.standard_normal(img.shape)
# sigma_hat = estimate_sigma(img, channel_axis=None)
# print('done!')

img = img_as_float(skimage.data.camera())
sigma = 0.1
rng = np.random.default_rng()
img = img + sigma * rng.standard_normal(img.shape)
sigma_hat = estimate_sigma(img, channel_axis=None)
print('done!')