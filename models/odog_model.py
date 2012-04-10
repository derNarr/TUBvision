#!/usr/bin/python
# -*- coding: latin-1 -*-
"""Implementation of the oriented difference of gaussians (ODOG) brightness
model by Blakeslee and McCourt"""

import numpy as np
from scipy.signal import fftconvolve

from utils import degrees_to_pixels, pad_array

class OdogModel(object):
    """
    Represents an oriented difference of gaussians model for lightness
    perception. The implementation follows publications by Blakeslee and
    McCourt (1997, 1999, 2003)
    """
    def __init__(self,
                 spatial_scales=3. / 2. ** np.arange(-2,5), 
                 orientations=np.arange(0,180,30),
                 pixels_per_degree=30,
                 weights_slope=.1):
        """
        Create an ODOG model instance.

        Parameters
        ----------
        spatial_scales : list of numbers, optional
                         the spatial frequencies of the filters. Each number
                         corresponds to one DOG filter and specifies the
                         center size, i.e. the distance from zero-crossing to
                         zero-crossing, in degrees of visual angle. Default is
                         an octave range from 12 to 3/16.
        orientations : list of numbers, optional
                       the orientations of the filters in degrees. 0 degrees is
                       a vertical filter, 90 degrees is a horizontal one.
                       Default is 30deg steps from 0 to 150.
        pixels_per_degree : number, optional
                            specifies how many pixels fit within one degree of
                            visual angle in the experimental setup. Default is
                            30.
        weights_slope : number, optional
                        the slope of the log-log function that relates a filter
                        spatial frequency to its weight in the summation.
                        Default is 0.1.
        """
        # determine standard deviation from zero_crossing distance,
        # assuming that the surround sigma is 2 times center sigma.
        center_sigmas = np.array(spatial_scales)
        center_sigmas = (center_sigmas / 2.) / (2 * np.sqrt(2 * np.log(2) / 3))
        center_sigmas = degrees_to_pixels(center_sigmas, pixels_per_degree)

        self.multiscale_filters = []
        for angle in orientations:
            scale_filters = []
            for center_sigma in center_sigmas:
                scale_filters.append(
                    difference_of_gaussians((center_sigma, 2 * center_sigma),
                                            center_sigma, angle))
            self.multiscale_filters.append(scale_filters)

        self.scale_weights = np.exp(weights_slope * np.log(spatial_scales))
        self.spatial_scales = spatial_scales
        self.orientations = orientations

    def evaluate(self, image, return_detailed=False):
        """
        Apply the model to an input image, represented as a 2D numpy array.
        """
        # use zero padding if there is no constant border value
        if not border_is_constant(image):
            pad_value = 0
        else:
            pad_value = image[0,0]
        # compute filter output for every orientation
        orientation_output = np.empty((image.shape[0], image.shape[1],
                                        len(self.multiscale_filters)))
        for i, multiscale_filter in enumerate(self.multiscale_filters):
            scale_output = np.empty((image.shape[0], image.shape[1],
                                        len(self.spatial_scales)))
            # convolve filters at all spatial scales, within one orientation,
            # with the image
            for j, kernel in enumerate(multiscale_filter):
                y_padding = (kernel.shape[0] + 1) / 2
                x_padding = (kernel.shape[1] + 1) / 2
                tmp_img = pad_array(image, np.array(((y_padding, y_padding),
                    (x_padding, x_padding))), pad_value)
                scale_output[:,:,j] = fftconvolve(tmp_img, kernel, 'same')[
                                    y_padding:tmp_img.shape[0] - y_padding,
                                    x_padding:tmp_img.shape[1] - x_padding]
            # compute the weighted sum over different spatial scales
            orientation_output[:,:,i] = np.dot(scale_output, self.scale_weights)
            # normalize filter response within each orientation with its std
            normalization = 1. / orientation_output[:,:,i].std()
            # set filters with practically no signal to 0 (rather arbitrary)
            if normalization > 1e10:
                normalization = 0
            orientation_output[:,:,i] *= normalization
        if return_detailed:
            return (orientation_output.sum(2), orientation_output)
        return orientation_output.sum(2)


def difference_of_gaussians(sigma_y, sigma_x, angle=0, size=None):
    """
    Compute a 2D difference of Gaussians kernel.

    Parameters
    ----------
    sigma_y : number or tuple of 2 numbers
              The standard deviations of the two Gaussians along the vertical
              if angle is 0, or along a line angle degrees from the vertical in
              the clockwise direction. If only a single number is given, it is
              used for both Gaussians.
    sigma_x : number or tuple of 2 numbers
              Same as sigma_y, only for the orthogonal direction.
    angle : number or tuple of two numbers, optional
            The rotation angles of the two Gaussians in degrees. If only a
            single number is given, it is used for both Gaussians. Default is
            0.
    size : tuple of two ints, optional
           the shape of the output. The default is chosen such that the output
           array is large enough to contain the kernel up to 5 standard
           deviations in each direction.
    """
    if not isinstance(sigma_y, (tuple, list)):
        sigma_y = (sigma_y, sigma_y)
    if not isinstance(sigma_x, (tuple, list)):
        sigma_x = (sigma_x, sigma_x)
    if not isinstance(angle, (tuple, list)):
        angle = (angle, angle)

    outer_gaussian = gaussian(sigma_y[1], sigma_x[1], angle[1], size)
    inner_gaussian = gaussian(sigma_y[0], sigma_x[0], angle[0],
                                outer_gaussian.shape)
    return inner_gaussian - outer_gaussian

def gaussian(sigma_y, sigma_x, angle=0, size=None):
    """
    compute a two-dimensional Gaussian kernel.

    Parameters
    ----------
    sigma_y : number
              the standard deviation of an unrotated kernel along the vertical
    sigma_x : number
              standard deviation of an unrotated kernel along the horizontal
    angle : number, optional
            the rotation angle of the kernel in the clockwise direction in
            degrees. Default is 0.
    size : tupel of 2 ints, optional
           the shape of the output. The default is chosen such that the output
           array is large enough to contain the kernel up to 5 standard
           deviations in each direction.

    Returns
    -------
    output : 2D numpy array
    """

    # compute the covariance matrix of the rotated multivariate gaussian
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    sigma = np.dot(np.dot(R, np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])), 
                    R.T)

    if size is None:
        size = (np.ceil(5 * sigma[1,1] ** .5), np.ceil(5 * sigma[0,0] ** .5))

    # create a grid on which to evaluate the multivariate gaussian pdf formula
    (Y,X) = np.ogrid[-(size[0] - 1) / 2. : size[0] / 2.,
                     -(size[1] - 1) / 2. : size[1] / 2.]
    # apply the gaussian pdf formula to every point in the grid
    return 1 / (np.linalg.det(sigma) ** .5 * 2 * np.pi) * \
            np.exp(-.5 / (1 - np.prod(sigma[::-1,:].diagonal()) /
                              np.prod(sigma.diagonal()))  * 
                    (X ** 2 / sigma[0,0] + Y ** 2 / sigma[1,1] -
                    2 * sigma[1,0] * X * Y / sigma[0,0] / sigma[1,1]))

def border_is_constant(arr):
    """
    Check if the border of an array has a constant value.
    """
    if len(arr.shape) != 2:
        raise ValueError('function only works for 2D arrays')
    border =  np.concatenate((arr[[0,-1],:].ravel(), arr[:,[0,-1]].ravel()))
    return len(np.unique(border)) == 1
