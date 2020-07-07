#include <iostream>
#include <opencv2/opencv.hpp>
#include <registration.hpp>

using namespace std;
using namespace cv;
using namespace registration;
using namespace registration::fmt;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "need two images : reference image and sensed image." << std::endl;
    return -1;
  }

  // Read reference image
  cv::Mat im0 = cv::imread(argv[1]);
  if (im0.empty()) {
    std::cout << "Could not open or find the reference image" << std::endl;
    return -1;
  }

  // Read image to be aligned
  cv::Mat im1 = cv::imread(argv[2]);
  if (im1.empty()) {
    std::cout << "Could not open or find the sensed image" << std::endl;
    return -1;
  }

  // Preprocess images
  if (im0.channels() == 3) {
    cvtColor(im0, im0, cv::COLOR_BGR2GRAY);
  }
  if (im1.channels() == 3) {
    cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
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
  cv::Mat transform = fourierMellinTransform(im0, im1);
  std::cout << "Translation [x, y] : " << extractTranslationFromAffine(transform) << std::endl;

  // Warp
  Mat imReg;
  warpAffine(im1, imReg, transform, im0.size());

  // Display results
  cv::namedWindow("Reference image", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Reference image", 300, 300);
  cv::moveWindow("Reference image", 0, 10);
  cv::imshow("Reference image", im0);

  cv::namedWindow("Sensed image", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Sensed image", 300, 300);
  cv::moveWindow("Sensed image", 400, 10);
  cv::imshow("Sensed image", im1);

  cv::namedWindow("Registered image", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Registered image", 300, 300);
  cv::moveWindow("Registered image", 800, 10);
  cv::imshow("Registered image", imReg);

  cv::waitKey(0);
}
