/**
 * Deriche edge detector (often referred to as Canny-Deriche detector) is an edge detection operator developed by Rachid Deriche in 1987.
 * It's a multistep algorithm used to obtain an optimal result of edge detection in a discrete two-dimensional image.
 * This algorithm is based on John F. Canny's work related to the edge detection (Canny's edge detector) and his criteria for optimal edge detection:
 *  - Detection quality – all existing edges should be marked and no false detection should occur.
 *  - Accuracy - the marked edges should be as close to the edges in the real image as possible.
 *  - Unambiguity - a given edge in the image should only be marked once. No multiple responses to one edge in the real image should occur.
 *
 * Deriche edge detector, like Canny edge detector, consists of the following 4 steps:
 *  1 - Smoothing
 *  2 - Calculation of magnitude and gradient direction
 *  3 - Non-maximum suppression
 *  4 - Hysteresis thresholding (using two thresholds)
 * The essential difference is in the implementation of the first two steps of the algorithm.
 * Unlike the Canny edge detector, Deriche edge detector uses the IIR Deriche filter.
 *
 * Using the IIR filter makes sense especially in cases where the processed image is noisy or a large amount of smoothing is required
 * (which leads to large convolution kernel for FIR filter). In these cases, the Deriche detector has considerable advantage over the Canny detector,
 * because it is able to process images in a short constant time independent of the desired amount of smoothing.
 */

#include "filters.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int DericheGradient(const cv::Mat &src, cv::Mat &dest, float gamma) {

  // Prefilter the image before applying the derivation operator (which naturally amplify the high-frequency noise)
  if (DericheBlur(src, dest, gamma))
    return -1;

  // Compute an image gradient (that is, the partial derivatives of the image along the x and y directions and add these two gradient)
  cv::Mat gx, gy, agx, agy;
  cv::Sobel(dest, gx, CV_32F, 1, 0, 1);
  cv::Sobel(dest, gy, CV_32F, 0, 1, 1);
  convertScaleAbs(gx, agx); // Convert 8 bits unsigned
  convertScaleAbs(gy, agy); // Convert 8 bits unsigned
  addWeighted(agx, .5, agy, .5, 0, dest);
  cv::normalize(dest, dest, 0, 255, cv::NORM_MINMAX);
  return 0;
}
