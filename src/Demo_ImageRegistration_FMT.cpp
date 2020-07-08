#include "iostream"
#include "opencv2/opencv.hpp"
#include "registration.hpp"

using namespace std;
using namespace cv;
using namespace registration;
using namespace registration::fmt;

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "need two images : reference image and sensed image." << endl;
    return -1;
  }

  // Read reference image
  Mat im0 = imread(argv[1]);
  if (im0.empty()) {
    cout << "Could not open or find the reference image" << endl;
    return -1;
  }

  // Read image to be aligned
  Mat im1 = imread(argv[2]);
  if (im1.empty()) {
    cout << "Could not open or find the sensed image" << endl;
    return -1;
  }

  // Preprocess images
  if (im0.channels() == 3) {
    cvtColor(im0, im0, COLOR_BGR2GRAY);
  }
  if (im1.channels() == 3) {
    cvtColor(im1, im1, COLOR_BGR2GRAY);
  }

  if (im0.type() == CV_8UC1) {
    im0.convertTo(im0, CV_64FC1, 1.0 / 255.0);
  }
  if (im1.type() == CV_8UC1) {
    im1.convertTo(im1, CV_64FC1, 1.0 / 255.0);
  }
  if (im0.type() == CV_32FC1) {
    im0.convertTo(im0, CV_64FC1);
  }
  if (im1.type() == CV_32FC1) {
    im1.convertTo(im1, CV_64FC1);
  }

  // Compute translation
  Mat transform = fourierMellinTransform(im0, im1);
  cout << "Translation [x, y] : " << extractTranslationFromAffine(transform) << endl;

  // Warp
  Mat imReg;
  warpAffine(im1, imReg, transform, im0.size());

  // Display results
  namedWindow("Reference image", WINDOW_GUI_NORMAL);
  resizeWindow("Reference image", 300, 300);
  moveWindow("Reference image", 0, 10);
  imshow("Reference image", im0);

  namedWindow("Sensed image", WINDOW_GUI_NORMAL);
  resizeWindow("Sensed image", 300, 300);
  moveWindow("Sensed image", 400, 10);
  imshow("Sensed image", im1);

  namedWindow("Registered image", WINDOW_GUI_NORMAL);
  resizeWindow("Registered image", 300, 300);
  moveWindow("Registered image", 800, 10);
  imshow("Registered image", imReg);

  waitKey(0);
}
