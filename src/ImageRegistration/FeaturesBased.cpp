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

  namespace {
    /**
     * Detect keypoints and compute descriptors using ORB
     * ORB is a fast binary descriptor based on the combination of the FAST keypoint detector and the BRIEF descriptor.
     * It is rotation invariant and robust to noise.

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
     * Detect keypoints and compute descriptors using AKAZE
     * KAZE(Accelerated-KAZE) is a sped-up version of KAZE.
     * It presents a fast multiscale feature detection and description approach for non-linear scale-spaces.
     * It is both scale and rotation invariant.
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
     * Brute-force matcher to find 2-nn matches with Hamming distance
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

  void featuresBasedRegistration(const Mat &im1, const Mat &im2, Mat &im1Reg, Mat &h, Mat &imMatches, FeaturesDescription algoToUse) {

    // Convert images to work on grayscale images
    Mat im1Gray, im2Gray;
    cvtColor(im1, im1Gray, COLOR_RGB2GRAY);
    cvtColor(im2, im2Gray, COLOR_RGB2GRAY);

    // 1 & 2 - Detect keypoints and compute descriptors
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    if (algoToUse == AKAZE_ALGO) {
      AKAZEFeatureDescription(im1Gray, im2Gray, keypoints1, keypoints2, descriptors1, descriptors2);
    } else {
      ORBFeatureDescription(im1Gray, im2Gray, keypoints1, keypoints2, descriptors1, descriptors2);
    }
    // 3- Match features
    std::vector<DMatch> matches;
    BruteForceHammingMatcher(descriptors1, descriptors2, matches);
    // sort matches by score
    std::sort(matches.begin(), matches.end());
    // remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());
    // extract location of good matches
    std::vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
      points1.push_back(keypoints1[matches[i].queryIdx].pt);
      points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // 3 - Estimate homography transformation using RANSAC and use it to warp image
    h = findHomography(points1, points2, RANSAC);
    warpPerspective(im1, im1Reg, h, im2.size());

    // OPTIONAL : store top matches (vizualization purpose)
    drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
  }

} // namespace registration
