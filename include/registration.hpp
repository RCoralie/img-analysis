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

  /**
   * ECC image alignment algorithm  (Alignment using Enhanced Correlation Coefficient Maximization)
   * @param im1         - reference image
   * @param im2         - sensed image
   * @param warp_matrix - the 2x3 or 3x3 warp matrix depending on the motion model.
   * @param warp_mode   - the warp model, depending on the motion model between images
   * @param warp_image  - the warped image.
   */
  void ECCRegistration(const cv::Mat &im1, const cv::Mat &im2, cv::Mat &warp_matrix, cv::Mat &warp_img, int warp_mode);

} // namespace registration

#endif
