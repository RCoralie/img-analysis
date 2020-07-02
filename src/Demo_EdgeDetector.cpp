#include "edgedetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

bool update = true;

//----------------------------------------------------------------------------//
void on_trackbar(int, void *) { update = true; }

//----------------------------------------------------------------------------//
int main(int argc, const char **argv) {
  if (argc < 2) {
    std::cout << "need one image." << std::endl;
    return -1;
  }

  cv::Mat imgSrc, imgFiltered, imgGray;

  imgSrc = cv::imread(argv[1]);
  if (imgSrc.empty()) {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  cv::cvtColor(imgSrc, imgGray, cv::COLOR_RGB2GRAY);

  cv::namedWindow("Canny-Deriche", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Canny-Deriche", 600, 600);
  cv::moveWindow("Canny-Deriche", 100, 100);

  int gamma;
  cv::createTrackbar("gamma", "Canny-Deriche", &gamma, 30, &on_trackbar);
  cv::setTrackbarPos("gamma", "Canny-Deriche", 5);

  while (true) {
    if (update) {
      DericheGradient(imgGray, imgFiltered, gamma / 10);
      cv::imshow("Canny-Deriche", imgFiltered);
    }
    update = false;
    if (cv::waitKey(1) == 27)
      exit(0);
  }

  return 0;
}
