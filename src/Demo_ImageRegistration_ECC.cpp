#include "iostream"
#include "opencv2/opencv.hpp"
#include "registration.hpp"

using namespace cv;
using namespace std;
using namespace registration;
using namespace registration::corr;

int main(int argc, char **argv) {
  // Read the images to be aligned
  if (argc < 3) {
    cout << "need two images (reference image and sensed image) with a motion euclidean." << endl;
    return -1;
  }

  // Read reference image
  Mat im1 = imread(argv[1]);
  if (im1.empty()) {
    cout << "Could not open or find the reference image" << endl;
    return -1;
  }

  // Read image to be aligned
  Mat im2 = imread(argv[2]);
  if (im2.empty()) {
    cout << "Could not open or find the sensed image" << endl;
    return -1;
  }

  Mat warp_matrix = enhancedCorrelationCoefficientMaximization(im1, im2, MOTION_TRANSLATION);
  Mat warp_img = imgRegistration(im1, im2, warp_matrix);

  // Show final result
  imshow("Ref image", im1);
  imshow("Sensed image", im2);
  imshow("Sensed image aligned", warp_img);
  waitKey(0);
}
