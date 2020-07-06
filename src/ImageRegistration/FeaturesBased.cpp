/**
 * Since the early 2000s, image registration has mostly used traditional feature-based approaches.
 * These approaches are based on three steps: Keypoint Detection and Feature Description, Feature Matching, and Image Warping.
 * In brief, we select points of interest in both images, associate each point of interest in the reference image to its equivalent in the sensed
 * image and transform the sensed image so that both images are aligned.
 *
 * AKAWE and ORB are efficient and free alternatives to SIFT, based on the same process :
 *     - Detect and describe keypoints
 *     - Match them using matcher
 *     - Estimate homography transformation using RANSAC
 *     - Apply homography transformation
 */

#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "registration.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

namespace registration {

  namespace featuresbased {

    namespace {
      /**
       * Detect keypoints and compute descriptors using ORB.
       *
       * @param im1          - reference image
       * @param im2          - sensed image
       * @param kpts1        - keypointsz of the reference image
       * @param kpts2        - keypoints of the sensed image
       * @param descriptors  - descriptors of the reference image
       * @param descriptors2 - descriptors of the sensed image
       */
      void ORBFeatureDescription(const Mat &im1, const Mat &im2, vector<KeyPoint> &kpts1, vector<KeyPoint> &kpts2, Mat &descriptors1,
                                 Mat &descriptors2) {
        Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
        orb->detectAndCompute(im1, Mat(), kpts1, descriptors1);
        orb->detectAndCompute(im2, Mat(), kpts2, descriptors2);
      }

      /**
       * Detect keypoints and compute descriptors using AKAZE.
       *
       * @param im1          - reference image
       * @param im2          - sensed image
       * @param kpts1        - keypointsz of the reference image
       * @param kpts2        - keypoints of the sensed image
       * @param descriptors  - descriptors of the reference image
       * @param descriptors2 - descriptors of the sensed image
       */
      void AKAZEFeatureDescription(const Mat &im1, const Mat &im2, vector<KeyPoint> &kpts1, vector<KeyPoint> &kpts2, Mat &descriptors1,
                                   Mat &descriptors2) {
        Ptr<Feature2D> akaze = AKAZE::create();
        akaze->detectAndCompute(im1, noArray(), kpts1, descriptors1); // since we don't need the mask parameter, noArray() is used.
        akaze->detectAndCompute(im2, noArray(), kpts2, descriptors2); // since we don't need the mask parameter, noArray() is used.
      }

      /**
       * Feature-matching performed by using Hamming distance for binary descriptors (AKAZE, ORB, BRISK).
       *
       * @param descriptors1 - [description]
       * @param descriptors2 - [description]
       * @param matches      - [description]
       */
      void BruteForceHammingMatcher(const Mat &descriptors1, const Mat &descriptors2, vector<DMatch> &matches) {
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors1, descriptors2, matches, Mat());
      }

    } // namespace

    // ---------------------------------------------------------------------------
    MatchFeatures findMatchFeatures(const Mat &im1, const Mat &im2, FeatureDetectorDescriptor algoToUse) {

      // Convert images to work on grayscale images
      Mat im1Gray, im2Gray;
      cvtColor(im1, im1Gray, COLOR_RGB2GRAY);
      cvtColor(im2, im2Gray, COLOR_RGB2GRAY);

      // 1 & 2 - Detect keypoints and compute descriptors
      struct ImgFeatures refImg = {vector<Point2f>(), vector<KeyPoint>(), Mat()};
      struct ImgFeatures sensedImg = {vector<Point2f>(), vector<KeyPoint>(), Mat()};
      if (algoToUse == AKAZE_ALGO) {
        AKAZEFeatureDescription(im1Gray, im2Gray, refImg.keypoints, sensedImg.keypoints, refImg.descriptors, sensedImg.descriptors);
      } else {
        ORBFeatureDescription(im1Gray, im2Gray, refImg.keypoints, sensedImg.keypoints, refImg.descriptors, sensedImg.descriptors);
      }
      // 3- Match features
      std::vector<DMatch> matches;
      BruteForceHammingMatcher(refImg.descriptors, sensedImg.descriptors, matches);
      // sort matches by score
      std::sort(matches.begin(), matches.end());
      // remove not so good matches
      const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
      matches.erase(matches.begin() + numGoodMatches, matches.end());
      // extract location of good matches
      for (size_t i = 0; i < matches.size(); i++) {
        refImg.bestMatchPts.push_back(refImg.keypoints[matches[i].queryIdx].pt);
        sensedImg.bestMatchPts.push_back(sensedImg.keypoints[matches[i].trainIdx].pt);
      }

      struct MatchFeatures matchFeatures = {refImg, sensedImg, Mat()};

      // OPTIONAL : store top matches (vizualization purpose)
      drawMatches(im1, refImg.keypoints, im2, sensedImg.keypoints, matches, matchFeatures.imgOfMatches);

      return matchFeatures;
    }

    // ---------------------------------------------------------------------------
    Mat findTransformationMatrix(const Mat &im1, const Mat &im2, FeatureDetectorDescriptor algoToUse, MotionModel model) {

      // 1 & 2 & 3 - Detect keypoints, compute descriptors and match features
      MatchFeatures matchFeatures = findMatchFeatures(im1, im2, algoToUse);

      // 4 - Estimate motion model transformation using robust method (RANSAC-based robust method or Least-Median robust method RANSAC)
      switch (model) {
      case AFFINE:
        return estimateAffine2D(matchFeatures.sensedImgFeatures.bestMatchPts, matchFeatures.refImgFeatures.bestMatchPts);
      case AFFINE_PARTIAL:
        return estimateAffinePartial2D(matchFeatures.sensedImgFeatures.bestMatchPts, matchFeatures.refImgFeatures.bestMatchPts);
      default:
        return findHomography(matchFeatures.sensedImgFeatures.bestMatchPts, matchFeatures.refImgFeatures.bestMatchPts, RANSAC);
      }
    }

    // ---------------------------------------------------------------------------
    Mat featuresBasedRegistration(const Mat &im1, const Mat &im2, FeatureDetectorDescriptor algoToUse, MotionModel model) {

      // Estimate motion model transformation
      Mat transformation = findTransformationMatrix(im1, im2, algoToUse, model);

      // Use motion model to warp sensed image
      Mat imgRegistered;
      if (model == AFFINE || model == AFFINE_PARTIAL) {
        warpAffine(im2, imgRegistered, transformation, im1.size());
      } else {
        warpPerspective(im2, imgRegistered, transformation, im1.size());
      }
      return imgRegistered;
    }

  } // namespace featuresbased

} // namespace registration
