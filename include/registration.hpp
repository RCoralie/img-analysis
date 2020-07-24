#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"

namespace registration {

  /**
   * Align an image according to a transformation matrix
   *
   * @param ref_ims           - reference image
   * @param sensed_img        - image to align
   * @param transformation    - transformation matrix
   *
   * @return the sensed image registered according to the transform matrix used
   */
  cv::Mat imgRegistration(const cv::Mat &ref_img, const cv::Mat &sensed_img, const cv::Mat &transformation);

  namespace featuresbased {

    /**
     * Configuration parameters for the features based registration.
     * This configuration provides default parameters
     */
    struct FBConfig {

      // Feature-detector-descriptor used in feature-matching
      enum FeatureDetectorDescriptor {
        ORB_ALGO,  // most efficient feature-detector-descriptor with least computational cost.
        AKAZE_ALGO // computationally efficient than SIFT, SURF and KAZE but expensive than ORB and BRISK.
      };

      // Spatial relationships between the reference and sensed images
      enum MotionModel {
        HOMOGRAPHY,    // perspective transformation between two planes (two 2D point sets).
        AFFINE,        // optimal 2D affine transformation between two 2D point sets.
        AFFINE_PARTIAL // optimal limited affine transformation with 4 degrees of freedom between two 2D point sets.
      };

      // Feature matching strategy applied for rejecting outliers and fitting the transformation models
      enum featuresMatching {
        RANSAC_METHOD, // can handle practically any ratio of outliers but need a threshold to distinguish inliers from outliers.
        LMEDS_METHOD   // Ldoes not need any threshold but it works correctly only when there are more than 50% of inliers.
      };

      FeatureDetectorDescriptor detectorDescriptor = ORB_ALGO;
      MotionModel model = HOMOGRAPHY;
      featuresMatching featuresMatching = RANSAC_METHOD;
      int maxFeatures = 500;
      float GoodMatchPercent = 0.15f;
    };

    /**
     * Represents matches between reference and sensed images
     */
    struct MatchFeatures {

      /**
       * Represents detected keypoints and computed descriptors on an image and its associated feature-matching points.
       */
      struct ImgFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<cv::Point2f> bestMatchPts;
      };

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
     * Features based registration : compute the transformation matrix between reference and sensed images
     *
     * @param im1       - reference image
     * @param im2       - sensed image
     * @param config    - parameters to use for the features based registration
     *
     * @return the transform matrix, according to the motion model used
     */
    cv::Mat featuresBasedMethod(const cv::Mat &im1, const cv::Mat &im2, FBConfig config = FBConfig());

  } // namespace featuresbased

  namespace fmt {

    /**
     * Fourier-Mellin based registration : compute the transformation matrix between reference and sensed images
     *
     * @param  ref_img    - ref image
     * @param  sensed_img - sensed image
     *
     * @return the affine transformation matrice
     */
    cv::Mat fourierMellinTransform(const cv::Mat &ref_img, const cv::Mat &sensed_img);

  } // namespace fmt

  namespace corr {
    /**
     * ECC image alignment algorithm  (Alignment using Enhanced Correlation Coefficient Maximization) :
     * compute the transformation matrix between reference and sensed images
     *
     * @param im1         - reference image
     * @param im2         - sensed image
     * @param warp_model  - the warp model, depending on the motion model between images :
     *                      cv::MOTION_HOMOGRAPHY
     *                      cv::MOTION_EUCLIDEAN
     *                      cv::MOTION_TRANSLATION
     *                      cv::MOTION_AFFINE
     *
     * @return the transform matrix, according to the motion model used
     */
    cv::Mat enhancedCorrelationCoefficientMaximization(const cv::Mat &im1, const cv::Mat &im2, int warp_model = cv::MOTION_HOMOGRAPHY);

  }; // namespace corr

} // namespace registration

#endif
