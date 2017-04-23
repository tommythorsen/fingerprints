# Fingerprint Image Enhancement
Implementation of fingerprint image enhancement filters. This repo contains two different filtering tools (`gabor` and `wahab`) and a library file for common utility functions (`utils.py`). Below is a general description of all three files. For more information, please read the source code.

## `wahab`
An executable script file that contains the code for applying the Wahab filter. It consists mainly of a function called `wahabKernel()` that creates a directional kernel for a given orientation, and a function called `wahabFilter()` that divides the image into cells, and convolves each cell with a directional kernel corresponding to the average orientation of the cell.

## `gabor`
An executable script file that contains the code for applying the Gabor filter. It contains the gaborKernel() function that creates a Gabor kernel for a given orientation and frequency. It contains two functions, `gaborFilter()` and `gaborFilterSubdivide()` processes the image by cell iteration or by area subdivision, respectively. They both
divide the image into smaller chunks, and convolve each chunk with a Gabor kernel corresponding to the average orientation in the chunk.

## `utils.py`
A Python file that is not meant to be invoked directly, but imported into other scripts. It contains a number of commonly useful functions for fingerprint image enhancement. The most important functions are:

### `convolve()`
A custom convolution function that allows us to convolve a whole image, or just a sub-area of an image.

### `findMask()`
Marks areas as good or bad, depending on the standard deviation of values within the area.

### `estimateOrientations()`
Creates an orientation field for an image, using a combination of the methods from [HWJ98] and [WCT98].

### `estimateFrequencies()`
Createsafrequencyfieldforanimage,usingthemethod from [HWJ98].

# References

## HWJ98
Hong, Lin; Wan, Yifei; Jain, Anil: Fingerprint image enhancement: Algorithm and performance evaluation. IEEE transactions on pattern analysis and machine intelligence, 20(8):777–789, 1998.

## WCT98
Wahab, A; Chin, SH; Tan, EC: Novel approach to automated fingerprint recognition. IEE Proceedings-Vision, Image and Signal Processing, 145(3):160–166, 1998.
