import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy

from PIL import Image
from io import BytesIO
import glob

image_list = []
for filename in glob.glob('Gaussian Mixtures/images/*/*.jpg'):
    im = np.array(Image.open(filename).getdata())
    rgb = [np.mean(im[:, 0] / 256.0), np.mean(im[:, 1] / 256.0), np.mean(im[:, 2] / 256.0)]
    image_list.append(rgb)