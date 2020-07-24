#include "tools.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace transform_mat {

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

    return homography(Rect(0, 0, 3, 2));
  }

  // ---------------------------------------------------------------------------
  Point2f extractTranslationFromAffine(const Mat &affine) {

    CV_Assert(affine.rows == 2 && affine.cols == 3);

    // The general rule for Matrices typenames in OpenCV is: CV_<bit_depth>(S|U|F)C<number_of_channels>
    // S = Signed integer
    // U = Unsigned integer
    // F = Float
    switch (affine.depth()) {
    case CV_8U:
      return Point2f(affine.at<uint8_t>(0, 2), affine.at<uint8_t>(1, 2));
    case CV_8S:
      return Point2f(affine.at<int8_t>(0, 2), affine.at<int8_t>(1, 2));
    case CV_16U:
      return Point2f(affine.at<uint16_t>(0, 2), affine.at<uint16_t>(1, 2));
    case CV_16S:
      return Point2f(affine.at<int16_t>(0, 2), affine.at<int16_t>(1, 2));
    case CV_32S:
      return Point2f(affine.at<int32_t>(0, 2), affine.at<int32_t>(1, 2));
    case CV_32F:
      return Point2f(affine.at<float>(0, 2), affine.at<float>(1, 2));
    case CV_64F:
      return Point2f(affine.at<double>(0, 2), affine.at<double>(1, 2));
    default:
      return Point2f(NAN, NAN);
    }
  } // namespace registration

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
}; // namespace transform_mat

//------------------------------------------------------------------------------
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

//----------------------------------------------------------------------------//
void hann(Mat &img) {
  CV_Assert(img.type() == CV_32FC1 || img.type() == CV_64FC1);

  Mat hann_window;
  createHanningWindow(hann_window, img.size(), img.type());
  img = img.mul(hann_window);
}

//----------------------------------------------------------------------------//
void erosion(Mat p_src, Mat &p_dst, int p_erosionType, int p_erosionSize) {
  Mat element = getStructuringElement(p_erosionType, Size(2 * p_erosionSize + 1, 2 * p_erosionSize + 1), Point(p_erosionSize, p_erosionSize));
  erode(p_src, p_dst, element);
}

//----------------------------------------------------------------------------//
void dilation(Mat p_src, Mat &p_dst, int p_dilationType, int p_dilationSize) {
  Mat element = getStructuringElement(p_dilationType, Size(2 * p_dilationSize + 1, 2 * p_dilationSize + 1), Point(p_dilationSize, p_dilationSize));
  dilate(p_src, p_dst, element);
}

//----------------------------------------------------------------------------//
void drawLines(Mat &p_img, vector<Vec4i> p_lines) {
  // Draw the lines
  for (size_t i = 0; i < p_lines.size(); i++) {
    Vec4i l = p_lines[i];
    line(p_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
  }
}

//----------------------------------------------------------------------------//
void drawLines(Mat &p_img, vector<Vec2f> p_lines) {
  // Draw the lines
  for (size_t i = 0; i < p_lines.size(); i++) {
    float rho = p_lines[i][0], theta = p_lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(p_img, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
  }
}

//----------------------------------------------------------------------------//
void localDispersion1D(Mat p_sig1D, Mat &p_result) {
  CV_Assert(p_sig1D.rows == 1 || p_sig1D.cols == 1);

  int deviation = 8;

  if (p_sig1D.rows > 1) {
    // Mirrored first and last values of 1D arrays to be able to calculate peaks on complete array without loss of information
    Mat sigRowsExtended;
    copyMakeBorder(p_sig1D, sigRowsExtended, deviation, deviation, 0, 0, BORDER_REFLECT);

    // Calculate the difference between local values ​​and their local neighbors to find significant amplitude changes and flatten the rest
    Mat significantRowsSignal(Mat::zeros(p_sig1D.size(), CV_32F));
    for (int i = deviation; i < sigRowsExtended.rows - deviation; i++) {
      float localMeanAtCenter =
          ((int)sigRowsExtended.at<uint8_t>(i, 0) + (int)sigRowsExtended.at<uint8_t>(i - 1, 0) + (int)sigRowsExtended.at<uint8_t>(i + 1, 0)) / 3;
      float localMeanAround = ((int)sigRowsExtended.at<uint8_t>(i - (deviation), 0) + (int)sigRowsExtended.at<uint8_t>(i - (deviation - 1), 0) +
                               (int)sigRowsExtended.at<uint8_t>(i - (deviation - 2), 0) + (int)sigRowsExtended.at<uint8_t>(i + (deviation), 0) +
                               (int)sigRowsExtended.at<uint8_t>(i + (deviation - 1), 0) + (int)sigRowsExtended.at<uint8_t>(i + (deviation - 2), 0)) /
                              6;
      float peakValue = localMeanAtCenter - localMeanAround;
      significantRowsSignal.at<float>(i - deviation, 0) = peakValue;
    }
    p_result = significantRowsSignal;
  }

  else if (p_sig1D.cols > 1) {
    // Mirrored first and last values of 1D arrays to be able to calculate peaks on complete array without loss of information
    Mat sigColumnsExtended;
    copyMakeBorder(p_sig1D, sigColumnsExtended, 0, 0, deviation, deviation, BORDER_REFLECT);

    // Calculate the difference between local values ​​and their local neighbors to find significant amplitude changes and flatten the rest
    Mat significantColsSignal(Mat::zeros(p_sig1D.size(), CV_32F));
    for (int i = deviation; i < sigColumnsExtended.cols - deviation; i++) {
      float localMeanAtCenter = ((int)sigColumnsExtended.at<uint8_t>(0, i) + (int)sigColumnsExtended.at<uint8_t>(0, i - 1) +
                                 (int)sigColumnsExtended.at<uint8_t>(0, i + 1)) /
                                3;
      float localMeanAround =
          ((int)sigColumnsExtended.at<uint8_t>(0, i - (deviation)) + (int)sigColumnsExtended.at<uint8_t>(0, i - (deviation - 1)) +
           (int)sigColumnsExtended.at<uint8_t>(0, i - (deviation - 2)) + (int)sigColumnsExtended.at<uint8_t>(0, i + (deviation)) +
           (int)sigColumnsExtended.at<uint8_t>(0, i + (deviation - 1)) + (int)sigColumnsExtended.at<uint8_t>(0, i + (deviation - 2))) /
          6;
      float peakValue = localMeanAtCenter - localMeanAround;
      significantColsSignal.at<float>(0, i - deviation) = peakValue;
    }
    p_result = significantColsSignal;
  }
}

//----------------------------------------------------------------------------//
void findPeaksIn1DSignal(Mat p_sig1D, Mat &p_result) {
  CV_Assert(p_sig1D.rows == 1 || p_sig1D.cols == 1);

  if (p_sig1D.rows > 1) {
    Mat significantRowsPeaks(Mat::zeros(p_sig1D.size(), CV_32F));
    for (int i = 1; i < p_sig1D.rows - 1; ++i) {
      float left = p_sig1D.at<float>(i - 1, 0);
      float cent = p_sig1D.at<float>(i, 0);
      float right = p_sig1D.at<float>(i + 1, 0);
      if (left < cent && right <= cent) {
        significantRowsPeaks.at<float>(i, 0) = cent;
      }
    }
    p_result = significantRowsPeaks;
  }

  else if (p_sig1D.cols > 1) {
    Mat significantColsPeaks(Mat::zeros(p_sig1D.size(), CV_32F));
    for (int i = 1; i < p_sig1D.cols - 1; ++i) {
      float left = p_sig1D.at<float>(0, i - 1);
      float cent = p_sig1D.at<float>(0, i);
      float right = p_sig1D.at<float>(0, i + 1);
      if (left < cent && right <= cent) {
        significantColsPeaks.at<float>(0, i) = cent;
      }
    }
    p_result = significantColsPeaks;
  }
}
