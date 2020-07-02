#ifndef DERICHE_HPP
#define DERICHE_HPP

#include "opencv2/core/core.hpp"

/**
 * This function make successively the two operations of prefiltering (Deriche smoothing filter) et the gradient computing (Sobel mask)
 *
 * @param  src       - input image array
 * @param  dest      - result image array
 * @param  gy        - gradient y calculated
 * @param  gamma     - filter parameter : defines the width of the filter, therefore the compromise between detection and localization.
 *                     The larger α, the more we localize precisely the outline.The smaller it is, the easier it is to detect the presence of the
 *                     edges. between 0 (no filtering) et 1 (maximal filtering).
 *
 * @return 0 if successful and -1 if not
 */
extern int DericheGradient(const cv::Mat &input, cv::Mat &output, float gamma);

#endif
