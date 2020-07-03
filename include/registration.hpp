#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include "opencv2/core/core.hpp"

namespace registration {

  enum FeaturesDescription { ORB_ALGO, AKAZE_ALGO };

  /**
   * Features based registation method using ORB or AKAZE algorithm to detect keypoints and compute descriptors.
   * @param im1       - reference image
   * @param im2       - sensed image
   * @param im1Reg    - image registered
   * @param h         - homography matrix
   * @param imMatches - image to vizualize matches between images
   * @param algoToUse - ORB_ALGO or AKAZE_ALGO
   */
  void featuresBasedRegistration(const cv::Mat &im1, const cv::Mat &im2, cv::Mat &im1Reg, cv::Mat &h, cv::Mat &imMatches,
                                 FeaturesDescription algoToUse);

} // namespace registration

#endif
