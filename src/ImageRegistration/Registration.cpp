#include "registration.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace registration {

  // ---------------------------------------------------------------------------
  Mat imgRegistration(const Mat &ref_img, const Mat &sensed_img, const Mat &transformation) {

    // Use transformation matrix to warp sensed image according to motion model
    Mat imgRegistered;
    if (transformation.rows == 3 && transformation.cols == 3) {
      warpPerspective(sensed_img, imgRegistered, transformation, ref_img.size());
    } else if (transformation.rows == 2 && transformation.cols == 3) {
      warpAffine(sensed_img, imgRegistered, transformation, ref_img.size());
    } else {
      return sensed_img;
    }
    return imgRegistered;
  }

  // ---------------------------------------------------------------------------
  Mat createHomographyMat(double alpha, double t_x, double t_y) {

    Mat homography = Mat::eye(3, 3, CV_64FC1);

    // TODO : hope there is no error in the signs...
    homography.at<double>(0, 0) = cos(CV_PI * alpha / 180);
    homography.at<double>(0, 1) = -sin(CV_PI * alpha / 180);
    homography.at<double>(1, 0) = sin(CV_PI * alpha / 180);
    homography.at<double>(1, 1) = cos(CV_PI * alpha / 180);
    homography.at<double>(0, 2) = t_x;
    homography.at<double>(1, 2) = t_y;

    return homography;
  }

  // ---------------------------------------------------------------------------
  Mat extractAffineFromHomography(const Mat &homography) {

    CV_Assert(homography.rows == 3 && homography.cols == 3);

    return homography(cv::Rect(0, 0, 3, 2));
  }

  // ---------------------------------------------------------------------------
  Point2f extractTranslationFromAffine(const Mat &affine) {

    CV_Assert(affine.rows == 2 && affine.cols == 3);

    double t_x = affine.at<double>(0, 2);
    double t_y = affine.at<double>(1, 2);
    return Point2f(t_x, t_y);
  }

  // ---------------------------------------------------------------------------
  Mat combineHomographies(const Mat &first, const Mat &second) {

    CV_Assert(first.rows == 3 && first.cols == 3);
    CV_Assert(second.rows == 3 && second.cols == 3);

    // combined by homograhy multiplication
    return first * second; // TODO: or second * first ?
  }

  // ---------------------------------------------------------------------------
  Mat combineAffines(const Mat &first, const Mat &second) {

    CV_Assert(first.rows == 2 && first.cols == 3);
    CV_Assert(second.rows == 2 && second.cols == 3);

    // create homographies to combine by homograhy multiplication

    Mat h1 = Mat::eye(3, 3, CV_64FC1);
    Mat h2 = Mat::eye(3, 3, CV_64FC1);

    Mat tmp_1, tmp_2, result_mat;
    first.convertTo(tmp_1, CV_64FC1);
    second.convertTo(tmp_2, CV_64FC1);

    tmp_1.row(0).copyTo(h1.row(0));
    tmp_1.row(1).copyTo(h1.row(1));

    tmp_2.row(1).copyTo(h2.row(1));
    tmp_2.row(0).copyTo(h2.row(0));

    result_mat = combineHomographies(h1, h2);

    h1.release();
    h2.release();
    tmp_1.release();
    tmp_2.release();

    return extractAffineFromHomography(result_mat);
  }

}; // namespace registration
