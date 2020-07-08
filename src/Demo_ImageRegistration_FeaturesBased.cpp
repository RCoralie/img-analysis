#include "iostream"
#include "opencv2/opencv.hpp"
#include "registration.hpp"

using namespace std;
using namespace cv;
using namespace registration;
using namespace registration::featuresbased;

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "need two images : reference image and sensed image." << endl;
    return -1;
  }

  // Read reference image
  Mat imReference = imread(argv[1]);
  if (imReference.empty()) {
    cout << "Could not open or find the reference image" << endl;
    return -1;
  }

  // Read image to be aligned
  Mat im = imread(argv[2]);
  if (im.empty()) {
    cout << "Could not open or find the sensed image" << endl;
    return -1;
  }

  // Print estimated homography
  cout << ">> Process matches ..." << endl;
  MatchFeatures matchFeatures = findMatchFeatures(imReference, im);

  // Print estimated homography
  cout << ">> Process transformation matrix ..." << endl;
  Mat h = featuresBasedMethod(imReference, im);
  cout << "Estimated motion model matrix : \n" << h << endl;

  // Align images
  cout << ">> Process image alignment ..." << endl;
  Mat imReg = imgRegistration(imReference, im, h);

  // Display results
  namedWindow("Reference image", WINDOW_GUI_NORMAL);
  resizeWindow("Reference image", 300, 300);
  moveWindow("Reference image", 0, 10);
  imshow("Reference image", imReference);

  namedWindow("Sensed image", WINDOW_GUI_NORMAL);
  resizeWindow("Sensed image", 300, 300);
  moveWindow("Sensed image", 400, 10);
  imshow("Sensed image", im);

  namedWindow("Registered image", WINDOW_GUI_NORMAL);
  resizeWindow("Registered image", 300, 300);
  moveWindow("Registered image", 800, 10);
  imshow("Registered image", imReg);

  namedWindow("Matches", WINDOW_GUI_NORMAL);
  resizeWindow("Matches", 1200, 500);
  moveWindow("Matches", 100, 400);
  imshow("Matches", matchFeatures.imgOfMatches);

  waitKey(0);

  // Store results
  // string outFilename("matches.jpg");
  // cout << "Saving matches vizualisation : " << outFilename << endl;
  // imwrite(outFilename, matchFeatures.imgOfMatches);
  //
  // string outFilename2("aligned.jpg");
  // cout << "Saving aligned image : " << outFilename2 << endl;
  // imwrite(outFilename2, imReg);
}
