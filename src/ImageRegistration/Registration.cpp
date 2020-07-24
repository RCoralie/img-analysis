#include "registration.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace registration {

  // ---------------------------------------------------------------------------
  Mat imgRegistration(const Mat &ref_img, const Mat &sensed_img, const Mat &transformation) {

    // Use transformation matrix to warp sensed image according to motion model
    Mat img_registered;
    if (transformation.rows == 3 && transformation.cols == 3) {
      warpPerspective(sensed_img, img_registered, transformation, ref_img.size());
    } else if (transformation.rows == 2 && transformation.cols == 3) {
      warpAffine(sensed_img, img_registered, transformation, ref_img.size());
    } else {
      return sensed_img;
    }
    return img_registered;
  }

}; // namespace registration
