# Gibbs Sampling for Image Denoising

Stanford EE 367 Final Project, Winter 2018

## Overview
This project applies Gibbs Sampling based on different Markov Random Fields (MRF) structures to solve the image denoising problems.
For more details, please refer to the [report](https://github.com/chang-yue/Gibbs-Gampling-for-Image-Denoising/blob/master/report.pdf).

There are three folders for different types of input images: binary, grayscale, and color.

## Running the code
To run the code, set hyper-parameters and noise level (flip_rate for binary image, or sigma for gray and color images) at the begining of each Python file, and specify the input img_name in the main function. Then, type `python2 gs_[input-type].py`.
For example, to see the effect of recovering a binary image einstein_equation.png, use 'einstein_equation' as img_name in the main function and then type `python2 gs_binary.py`.
