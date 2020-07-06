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
      ORB_ALGO,  // fast binary descriptor based on the combination of the FAST keypoint detector and the BRIEF descriptor.
      AKAZE_ALGO // fast binary descriptor based on a speed-up version of KAZE.
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
     * [findMatchFeatures description]
     * @param  im1       [description]
     * @param  im2       [description]
     * @param  algoToUse [description]
     * @return           [description]
     */
    MatchFeatures findMatchFeatures(const cv::Mat &im1, const cv::Mat &im2, FeatureDetectorDescriptor algoToUse);

    /**
     * Compute matches between reference and sensed images
     * @param im1       - reference image
     * @param im2       - sensed image
     * @param algoToUse - ORB_ALGO   :  ORB is the most efficient feature-detector-descriptor with least computational cost.
     *                    AKAZE_ALGO :  AKAZE is computationally efficient than SIFT, SURF and KAZE but expensive than ORB and BRISK.
     * @param model     - HOMOGRAPHY     :  estimates a perspective transformation between two planes (two 2D point sets).
     *                    AFFINE         :  estimates an optimal 2D affine transformation between two 2D point sets.
     *                    AFFINE_PARTIAL :  estimates an optimal limited affine transformation with 4 degrees of freedom between two 2D point sets.
     *
     * @return the transform matrix, according to the motion model used
     */
    cv::Mat findTransformationMatrix(const cv::Mat &im1, const cv::Mat &im2, FeatureDetectorDescriptor algoToUse, MotionModel model);

    /**
     * Features based registation method
     * @param im1       - reference image
     * @param im2       - sensed image
     * @param algoToUse - ORB_ALGO   :  ORB is the most efficient feature-detector-descriptor with least computational cost.
     *                    AKAZE_ALGO :  AKAZE is computationally efficient than SIFT, SURF and KAZE but expensive than ORB and BRISK.
     * @param model     - HOMOGRAPHY     :  estimates a perspective transformation between two planes (two 2D point sets).
     *                    AFFINE         :  estimates an optimal 2D affine transformation between two 2D point sets.
     *                    AFFINE_PARTIAL :  estimates an optimal limited affine transformation with 4 degrees of freedom between two 2D point sets.
     *
     * @return the sensed image registered according to the transform matrix used
     */
    cv::Mat featuresBasedRegistration(const cv::Mat &im1, const cv::Mat &im2, FeatureDetectorDescriptor algoToUse, MotionModel model);

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
