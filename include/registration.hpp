#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

namespace registration {

  namespace featuresbased {
    /**
     * Feature-detector-descriptor used in feature-matching
     */
    enum FeatureDetectorDescriptor {
      ORB_ALGO,  // most efficient feature-detector-descriptor with least computational cost.
      AKAZE_ALGO // computationally efficient than SIFT, SURF and KAZE but expensive than ORB and BRISK.
    };

    /**
     *  The spatial relationships between the reference and sensed images
     */
    enum MotionModel {
      HOMOGRAPHY,    // perspective transformation between two planes (two 2D point sets).
      AFFINE,        // optimal 2D affine transformation between two 2D point sets.
      AFFINE_PARTIAL // optimal limited affine transformation with 4 degrees of freedom between two 2D point sets.
    };

    /**
     * Feature matching strategy applied for rejecting outliers and fitting the transformation models
     */
    enum featureMatching {
      RANSAC_METHOD, // RANSAC robust method : can handle practically any ratio of outliers but need a threshold to distinguish inliers from outliers.
      LMEDS_METHOD   // Least-Median robust method : does not need any threshold but it works correctly only when there are more than 50% of inliers.
    };

    /**
     * Configuration parameters for the features based registration
     */
    struct FBConfig {
      FeatureDetectorDescriptor detectorDescriptor = ORB_ALGO;
      MotionModel model = HOMOGRAPHY;
      int featuresMatching = RANSAC_METHOD;
      int maxFeatures = 500;
      float GoodMatchPercent = 0.15f;
    };

    /**
     * Represents detected keypoints and computed descriptors on an image and its associated feature-matching points.
     */
    struct ImgFeatures {
      std::vector<cv::Point2f> bestMatchPts;
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
    };

    /**
     * Represents matches between reference and sensed images
     */
    struct MatchFeatures {
      ImgFeatures refImgFeatures;    // match features on reference image
      ImgFeatures sensedImgFeatures; // match features on sensed image
      cv::Mat imgOfMatches;          // matrix to vizualize matches between images
    };

    /**
     * Compute matches between reference and sensed images
     *
     * @param im1       - reference image
     * @param im2       - sensed image
     * @param config    - parameters to use for the features based registration
     *
     * @return the match features between reference and sensed images
     */
    MatchFeatures findMatchFeatures(const cv::Mat &im1, const cv::Mat &im2, FBConfig config = FBConfig());

    /**
     * Compute the transformation matrix between reference and sensed images
     *
     * @param im1       - reference image
     * @param im2       - sensed image
     * @param config    - parameters to use for the features based registration
     *
     * @return the transform matrix, according to the motion model used
     */
    cv::Mat findTransformationMatrix(const cv::Mat &im1, const cv::Mat &im2, FBConfig config = FBConfig());

    /**
     * Features based registation method
     *
     * @param im1       - reference image
     * @param im2       - sensed image
     * @param config    - parameters to use for the features based registration
     *
     * @return the sensed image registered according to the transform matrix used
     */
    cv::Mat featuresBasedRegistration(const cv::Mat &im1, const cv::Mat &im2, FBConfig config = FBConfig());

  } // namespace featuresbased

  /**
   * ECC image alignment algorithm  (Alignment using Enhanced Correlation Coefficient Maximization)
   *
   * @param im1         - reference image
   * @param im2         - sensed image
   * @param warp_matrix - the 2x3 or 3x3 warp matrix depending on the motion model.
   * @param warp_mode   - the warp model, depending on the motion model between images
   * @param warp_image  - the warped image.
   */
  void ECCRegistration(const cv::Mat &im1, const cv::Mat &im2, cv::Mat &warp_matrix, cv::Mat &warp_img, int warp_mode);

} // namespace registration

#endif
