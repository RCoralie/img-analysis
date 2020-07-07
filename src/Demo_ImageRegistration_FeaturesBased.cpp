#include "iostream"
#include "opencv2/opencv.hpp"
#include "registration.hpp"

using namespace std;
using namespace cv;
using namespace registration::featuresbased;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "need two images : reference image and sensed image." << std::endl;
    return -1;
  }

  // Read reference image
  cv::Mat imReference = cv::imread(argv[1]);
  if (imReference.empty()) {
    std::cout << "Could not open or find the reference image" << std::endl;
    return -1;
  }

  // Read image to be aligned
  cv::Mat im = cv::imread(argv[2]);
  if (im.empty()) {
    std::cout << "Could not open or find the sensed image" << std::endl;
    return -1;
  }

  // Print estimated homography
  cout << ">> Process matches ..." << endl;
  MatchFeatures matchFeatures = findMatchFeatures(imReference, im);

  // Print estimated homography
  cout << ">> Process transformation matrix ..." << endl;
  Mat h = findTransformationMatrix(imReference, im);
  cout << "Estimated motion model matrix : \n" << h << endl;

  // Align images
  cout << ">> Process image alignment ..." << endl;
  Mat imReg = featuresBasedRegistration(imReference, im);

  // Display results
  cv::namedWindow("Reference image", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Reference image", 300, 300);
  cv::moveWindow("Reference image", 0, 10);
  cv::imshow("Reference image", imReference);

  cv::namedWindow("Sensed image", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Sensed image", 300, 300);
  cv::moveWindow("Sensed image", 400, 10);
  cv::imshow("Sensed image", im);

  cv::namedWindow("Registered image", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Registered image", 300, 300);
  cv::moveWindow("Registered image", 800, 10);
  cv::imshow("Registered image", imReg);

  cv::namedWindow("Matches", cv::WINDOW_GUI_NORMAL);
  cv::resizeWindow("Matches", 1200, 500);
  cv::moveWindow("Matches", 100, 400);
  cv::imshow("Matches", matchFeatures.imgOfMatches);

  cv::waitKey(0);

  // Store results
  // string outFilename("matches.jpg");
  // cout << "Saving matches vizualisation : " << outFilename << endl;
  // imwrite(outFilename, matchFeatures.imgOfMatches);
  //
  // string outFilename2("aligned.jpg");
  // cout << "Saving aligned image : " << outFilename2 << endl;
  // imwrite(outFilename2, imReg);
}
