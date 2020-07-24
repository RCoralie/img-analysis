/**
 * The Deriche filter is a smoothing filter (low-pass).
 *
 * Deriche filter is very similar to Gaussian filter, but much simpler to implement (based on simple first order IIR filters).
 * Indeed, contrary to a gaussian filter that is often implemented using a FIR (finite response) filter,
 * and which complexity is directly dependant on the desired filtering level (standard deviation sigma),
 * for a first order IIR, which equation is: y[n] = α*x[n] + (1-α)*y[n-1], the complexity is constant and very limited (2 multiplications per pixel),
 * and the filtering level can be arbitrary modified through the "forget factor" α, also called gamma.
 *
 *  * The advantage of such a filter is that it can be adapted to the characteristics of the processed image using only one parameter : gamma.
 *  - If the value of α is small (usually between 0.25 and 0.5), it results in better detection.
 *  - On the other hand, better localization is achieved when the parameter has a higher value (around 2 or 3).
 *  - For most of the normal cases parameter value of around 1 is recommended.
 * Using the IIR filter makes sense especially in cases where the processed image is noisy or a large amount of smoothing is required
 * (which leads to large convolution kernel for FIR filter). In these cases, the Deriche detector has considerable advantage over the Canny detector,
 * because it is able to process images in a short constant time independent of the desired amount of smoothing.
 */

#include "filters.hpp"

/** Causal smoothing filter (Garcia Lorca operator) : y(n) = (1-γ)*x(n) + γ*y(n-1)
 * @param x      - input 1D signal
 * @param y      - filtered 1D signal
 * @param n      - size of the input signal
 * @param gamma  - expt(-α)
 * @param stride - the step
 */
template <typename T> static void causal_filter(const T *x, T *y, uint16_t n, float gamma, uint16_t stride) {
  float accu = x[0];
  for (uint16_t i = 0; i < n; i++) {
    accu = (1.0 - gamma) * x[i * stride] + gamma * accu;
    y[i * stride] = (T)accu;
  }
}

/** Anti-causal smoothing filter (Garcia Lorca operator) : y(n) = (1-γ)*x(n) + γ*y(n+1)
 * @param x      - input 1D signal
 * @param y      - filtered 1D signal
 * @param n      - size of the input signal
 * @param gamma  - expt(-α)
 * @param stride - the step
 */
template <typename T> static void anticausal_filter(const T *x, T *y, uint16_t n, float gamma, uint16_t stride) {
  float accu = x[(n - 1) * stride];
  for (signed short i = n - 1; i >= 0; i--) {
    accu = (1.0 - gamma) * x[i * stride] + gamma * accu;
    y[i * stride] = (T)accu;
  }
}

/**
 * Apply causal smoothing filter * 2.
 */
template <typename T> void dual_causal_filter(const T *x, T *y, uint16_t n, float gamma, uint16_t stride) {
  causal_filter<T>(x, y, n, gamma, stride);
  causal_filter<T>(x, y, n, gamma, stride);
}

/**
 * Apply anti-causal smoothing filter * 2.
 */
template <typename T> void dual_anticausal_filter(const T *x, T *y, uint16_t n, float gamma, uint16_t stride) {
  anticausal_filter<T>(x, y, n, gamma, stride);
  anticausal_filter<T>(x, y, n, gamma, stride);
}

/**
 * Approximation of the Deriche's one-dimensional filter by Garcia-Lorca operators :
 * Apply causal smoothing filter * 2 and anti-causal smoothing filter * 2.
 */
template <typename T> void GLSmoothing_Deriche_1D(const T *x, T *y, uint16_t n, float gamma, uint16_t stride) {
  causal_filter<T>(x, y, n, gamma, stride);
  anticausal_filter<T>(x, y, n, gamma, stride);
}

/**
 * Approximation of the Shen's one-dimensional filter by Garcia-Lorca operators :
 * Apply causal smoothing filter and anti-causal smoothing filter.
 */
template <typename T> void GLSmoothing_Shen_1D(const T *x, T *y, uint16_t n, float gamma, uint16_t stride) {
  causal_filter<T>(x, y, n, gamma, stride);
  anticausal_filter<T>(x, y, n, gamma, stride);
}

/**
 * Garcia Lorca operators are one-dimensional operators of order one, used in cascade :
 *   - causal filter : y(n) = (1-γ)*x(n) + γ*y(n-1)
 *   - anti-causal filter : y(n) = (1-γ)*x(n) + γ*y(n+1)
 *   - with γ = exp(-α)
 *  By applying this cascade filter once, we obtain a good approximation of Shen's one-dimensional filter.
 *  By applying it twice, we obtain a good approximation of the Deriche's one-dimensional filter.
 *  Two-dimensional filters are obtained by horizontal and vertical application of one-dimensional filters.
 *
 * @param input   - input two-dimensional signal
 * @param output  - filtered two-dimensional signal
 * @param gamma   - γ = exp(-α)
 */
template <typename T> void GLSmoothing_Deriche_2D(const cv::Mat &input, cv::Mat &output, float gamma) {

  uint16_t nbCols = input.cols;
  uint16_t nbRows = input.rows;
  uint16_t nbChannels = input.channels();

  // Transposition of the input signal
  cv::Mat filteredSignal = input.t();

  // Vertical filtering with causal smoothing filter * 2 and anti-causal smoothing filter * 2.
  for (auto c = 0; c < nbChannels; c++) {
    for (uint16_t x = 0u; x < nbCols; x++)
      GLSmoothing_Deriche_1D<T>(filteredSignal.ptr<T>(x) + c, filteredSignal.ptr<T>(x) + c, nbRows, gamma, nbChannels); // ligne x
  }

  // Transposition of the vertically filtered matrix
  output = filteredSignal.t();

  // Horizontal filtering with causal smoothing filter * 2 and anti-causal smoothing filter * 2.
  for (auto c = 0; c < nbChannels; c++) {
    for (uint16_t y = 0u; y < nbRows; y++)
      GLSmoothing_Deriche_1D<T>(output.ptr<T>(y) + c, output.ptr<T>(y) + c, nbCols, gamma, nbChannels);
  }
}

int DericheBlur(const cv::Mat &input, cv::Mat &output, float gamma) {
  switch (input.depth()) {
  case CV_32F:
    GLSmoothing_Deriche_2D<float>(input, output, gamma);
    break;
  case CV_8U:
    GLSmoothing_Deriche_2D<uint8_t>(input, output, gamma);
    break;
  case CV_16S:
    GLSmoothing_Deriche_2D<int16_t>(input, output, gamma);
    break;
  default:
    fprintf(stderr, "Deriche smoothing : Data type unsupported(%d).\n", input.depth());
    return -1;
  }
  return 0;
}
