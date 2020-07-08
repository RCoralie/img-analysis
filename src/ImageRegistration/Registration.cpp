#include "registration.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace registration {

  // ---------------------------------------------------------------------------
  Mat imgRegistration(const Mat &ref_img, const Mat &sensed_img, const Mat &transformation, MotionModel motion_model) {

    // Use transformation matrix to warp sensed image according to motion model
    Mat imgRegistered;
    if ((motion_model == MM_HOMOGRAPHY || motion_model == MM_UNKNOWN) && (transformation.rows == 3 && transformation.cols == 3)) {
      warpPerspective(sensed_img, imgRegistered, transformation, ref_img.size());
    } else if (motion_model != MM_HOMOGRAPHY && (transformation.rows == 2 && transformation.cols == 3)) {
      warpAffine(sensed_img, imgRegistered, transformation, ref_img.size());
    }
    return imgRegistered;
  }

}; // namespace registration
