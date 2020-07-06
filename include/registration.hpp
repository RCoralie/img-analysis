#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

namespace registration {

  enum FeaturesDescription { ORB_ALGO, AKAZE_ALGO };

  enum MotionModel { HOMOGRAPHY, AFFINE, AFFINE_PARTIAL };

  struct ImgFeatures {
    std::vector<cv::Point2f> bestMatchPts;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
  };

  struct MatchFeatures {
    ImgFeatures refImgFeatures;    // match features on reference image
    ImgFeatures sensedImgFeatures; // match features on sensed image
    cv::Mat imgOfMatches;          // image to vizualize matches between images
  };

  /**
   * [findMatchFeatures description]
   * @param  im1       [description]
   * @param  im2       [description]
   * @param  algoToUse [description]
   * @return           [description]
   */
  MatchFeatures findMatchFeatures(const cv::Mat &im1, const cv::Mat &im2, FeaturesDescription algoToUse);

  /**
   * Features based registation method using ORB or AKAZE algorithm to detect keypoints and compute descriptors.
   * @param im1       - reference image
   * @param im2       - sensed image
   * @param algoToUse - ORB_ALGO or AKAZE_ALGO
   * @param model     - HOMOGRAPHY :       estimates a perspective transformation between two planes (two 2D point sets).
   *                    AFFINE :           estimates an optimal 2D affine transformation between two 2D point sets.
   *                    AFFINE_PARTIAL :   estimates an optimal limited affine transformation with 4 degrees of freedom (limited to combinations of
   *                                       translation, rotation, and uniform scaling) between two 2D point sets.
   *
   * @return the transform matrix, according to the motion model used
   */
  cv::Mat findTransformationMatrix(const cv::Mat &im1, const cv::Mat &im2, FeaturesDescription algoToUse, MotionModel model);

  /**
   * Features based registation method using ORB or AKAZE algorithm to detect keypoints and compute descriptors.
   * @param im1       - reference image
   * @param im2       - sensed image
   * @param algoToUse - ORB_ALGO or AKAZE_ALGO
   * @param model     - HOMOGRAPHY :       estimates a perspective transformation between two planes (two 2D point sets).
   *                    AFFINE :           estimates an optimal 2D affine transformation between two 2D point sets.
   *                    AFFINE_PARTIAL :   estimates an optimal limited affine transformation with 4 degrees of freedom (limited to combinations of
   *                                       translation, rotation, and uniform scaling) between two 2D point sets.
   *
   * @return the sensed image registered according to the transform matrix used
   */
  cv::Mat featuresBasedRegistration(const cv::Mat &im1, const cv::Mat &im2, FeaturesDescription algoToUse, MotionModel model);

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
