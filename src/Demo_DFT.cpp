#include "fouriertransform.hpp"
#include "tools.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

bool update = true;

//----------------------------------------------------------------------------//
void transform(cv::Mat &img, int x, int y, int r, float s) {
  cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, x, 0, 1, y);
  cv::warpAffine(img, img, M, img.size());

  M = getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), 0, 1);
  cv::warpAffine(img, img, M, img.size());

  M = getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), 0, s);
  cv::warpAffine(img, img, M, img.size());

  M = getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), r, 1);
  cv::warpAffine(img, img, M, img.size());
}

//----------------------------------------------------------------------------//
void on_trackbar(int, void *) { update = true; }

//----------------------------------------------------------------------------//
void pipeline(cv::Mat &img, cv::Mat &dst, int x, int y, int r, float s) {
  transform(img, x, y, r, s);

  /// Apply the gaussian blur
  Mat bluredImg;
  GaussianBlur(img, bluredImg, Size(29, 29), 0, 0, BORDER_DEFAULT);

  /// Hanning window
  cv::Mat hanning;
  bluredImg.convertTo(hanning, CV_64FC1);
  hann(hanning);

  /// Compute FFT
  cv::Mat dft;
  DFT(hanning, dft);
  Mat fftMag;
  magnitudeSpectrum(dft, fftMag);
  Mat fftPhase;
  phaseSpectrum(dft, fftPhase);

  /// Compute IFFT
  Mat idft;
  IDFT(dft, idft);

  /// Display results
  cv::Mat input;
  hanning.convertTo(input, CV_8U);
  cv::imshow("input", input);

  cv::imshow("FFT Magnitude", fftMag);

  cv::imshow("FFT Phase", fftPhase);

  cv::Mat output;
  idft.convertTo(output, CV_8U);
  cv::imshow("IFFT", output);
}

//----------------------------------------------------------------------------//
void displayCommandsWindowAndResults(Mat &p_src) {
  cv::namedWindow("input", WINDOW_GUI_NORMAL);
  cv::resizeWindow("input", 620, 310);
  cv::moveWindow("input", 0, 0);

  cv::namedWindow("FFT Magnitude", WINDOW_GUI_NORMAL);
  cv::resizeWindow("FFT Magnitude", 620, 310);
  cv::moveWindow("FFT Magnitude", 320, 0);

  cv::namedWindow("FFT Phase", WINDOW_GUI_NORMAL);
  cv::resizeWindow("FFT Phase", 620, 310);
  cv::moveWindow("FFT Phase", 640, 0);

  cv::namedWindow("IFFT", WINDOW_GUI_NORMAL);
  cv::resizeWindow("IFFT", 620, 310);
  cv::moveWindow("IFFT", 960, 0);

  cv::namedWindow("commands", WINDOW_GUI_NORMAL);
  cv::resizeWindow("FFT Phase", 620, 620);
  cv::moveWindow("commands", 1920 / 2 - 150, 700);
  int translation_x = 100;
  cv::createTrackbar("translation x", "commands", &translation_x, 200, &on_trackbar);
  cv::setTrackbarPos("translation x", "commands", 100);
  int translation_y = 100;
  cv::createTrackbar("translation y", "commands", &translation_y, 200, &on_trackbar);
  cv::setTrackbarPos("translation y", "commands", 100);

  int rotation = 90;
  cv::createTrackbar("rotation", "commands", &rotation, 180, &on_trackbar);
  cv::setTrackbarPos("rotation", "commands", 90);

  int scale = 50;
  cv::createTrackbar("scale", "commands", &scale, 100, &on_trackbar);
  cv::setTrackbarPos("scale", "commands", 50);

  while (true) {
    if (update) {
      cv::Mat img_mod = p_src.clone();
      cv::Mat output;
      pipeline(img_mod, output, translation_x - 100, translation_y - 100, rotation - 90, float(scale) / 50);
    }
    update = false;
    if (cv::waitKey(1) == 27)
      exit(0);
  }
}

//----------------------------------------------------------------------------//
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "need one image." << std::endl;
    return -1;
  }

  Mat I = imread(argv[1], IMREAD_REDUCED_GRAYSCALE_2);
  if (I.empty()) {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  displayCommandsWindowAndResults(I);

  return 0;
}
