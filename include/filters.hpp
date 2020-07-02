#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "opencv2/core/core.hpp"

/**
 * Deriche smoothing by Garcia Lorca operators.
 *
 * @param input   - input two-dimensional signal
 * @param output  - filtered two-dimensional signal
 * @param gamma   - γ = exp(-α)
 *
 * @return 0 if successful and -1 if not
 */
int DericheBlur(const cv::Mat &input, cv::Mat &output, float gamma);

#endif
