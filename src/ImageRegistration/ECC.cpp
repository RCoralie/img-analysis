/**
 * ECC image alignment algorithm estimates the geometric transformation (warp) between the input and the template image and it returns the warped
 * input image which must be close to the template one, as it is shown below. The estimated transformation is the one that maximizes the correlation
 * coefficient between the template and the warped input image.
 */

#include "opencv2/opencv.hpp"
#include "registration.hpp"

using namespace std;
using namespace cv;

// Specify the number of iterations.
const int NB_ITERATIONS = 1000;

// Specify the threshold of the increment in the correlation coefficient between two iterations
const double TERMINATION_EPS = 1e-10;

namespace registration {

  namespace corr {

    Mat enhancedCorrelationCoefficientMaximization(const Mat &im1, const Mat &im2, int warp_mode) {

      // Convert images to work on grayscale images
      Mat im1Gray, im2Gray;
      cvtColor(im1, im1Gray, COLOR_RGB2GRAY);
      cvtColor(im2, im2Gray, COLOR_RGB2GRAY);

      // Initialize the matrix to identity
      Mat warp_matrix;
      if (warp_mode == MOTION_HOMOGRAPHY)
        warp_matrix = Mat::eye(3, 3, CV_32F);
      else

        warp_matrix = Mat::eye(2, 3, CV_32F);
      // Define termination criteria & run the ECC algorithm
      TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, NB_ITERATIONS, TERMINATION_EPS);
      findTransformECC(im2Gray, im1Gray, warp_matrix, warp_mode, criteria);

      return warp_matrix;
    }

  }; // namespace corr

}; // namespace registration
