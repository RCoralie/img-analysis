#include "edgedetect.hpp"
#include "registration.hpp"
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace registration;
using namespace registration::fmt;
using namespace registration::featuresbased;
using namespace registration::corr;
using namespace boost::program_options;

boost::regex number_regex("-?[0-9]+([\\.][0-9]+)?");

void callbackButton(int state, void *pointer) { printf("ok"); }

int main(int argc, char **argv) {

  // Declare the supported options.
  options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");
  desc.add_options()("correlation", value<vector<string>>()->multitoken(),
                     "image registration using the ECC method : [reference image path] [sensed image path]");
  desc.add_options()("fmt", value<vector<string>>()->multitoken(),
                     "image registration using the fourier-mellin transform method : [reference image path] [sensed image path]");
  desc.add_options()("orb", value<vector<string>>()->multitoken(),
                     "image registration using the features based ORB method method : [reference image path] [sensed image path]");
  desc.add_options()("akaze", value<vector<string>>()->multitoken(),
                     "image registration using the features based AKAZE method method : [reference image path] [sensed image path]");
  desc.add_options()("model", value<string>()->default_value("HOMOGRAPHY"),
                     "motion model used for registration : HOMOGRAPHY, AFFINE, RIGID, TRANSLATION");
  desc.add_options()("features-matching", value<string>()->default_value("RANSAC"),
                     "Feature matching strategy applied for features based methods : RANSAC, LMEDS");
  desc.add_options()("preprocess,P", value<float>()->default_value(0.5), "Canny-Deriche edge detector applied before registration.");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  // --------------------------------------------------------------------------
  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  // --------------------------------------------------------------------------
  vector<string> arg;
  if (vm.count("correlation")) {
    arg = vm["correlation"].as<vector<string>>();
  } else if (vm.count("fmt")) {
    arg = vm["fmt"].as<vector<string>>();
  } else if (vm.count("orb")) {
    arg = vm["orb"].as<vector<string>>();
  } else if (vm.count("akaze")) {
    arg = vm["akaze"].as<vector<string>>();
  } else {
    cout << "Expected an argument to specify image registration method : correlation, fmt, orb or akaze. (see --help for more details)" << endl;
    return -1;
  }
  if (arg.size() < 2) {
    cout << "Expected at least 2 arguments : [reference image path] [sensed image path]" << endl;
    return -1;
  }
  Mat ref_img = imread(arg[0]);
  if (ref_img.empty()) {
    cout << "Could not open or find the [reference image path]" << endl;
    return -1;
  }
  Mat sensed_img = imread(arg[1]);
  if (sensed_img.empty()) {
    cout << "Could not open or find the [sensed image path]" << endl;
    return -1;
  }

  Mat ref_img_preprocessed = ref_img.clone();
  Mat sensed_img_preprocessed = sensed_img.clone();
  if (vm.count("preprocess")) {
    cv::cvtColor(ref_img, ref_img_preprocessed, cv::COLOR_RGB2GRAY);
    cv::cvtColor(sensed_img, sensed_img_preprocessed, cv::COLOR_RGB2GRAY);
    DericheGradient(ref_img_preprocessed, ref_img_preprocessed, 0.5);
    DericheGradient(sensed_img_preprocessed, sensed_img_preprocessed, 0.5);
  }

  Mat warp_mat, warp_img, match_img;
  FBConfig config;
  config.model = FBConfig::HOMOGRAPHY;      // TODO by default
  int motion_model = cv::MOTION_HOMOGRAPHY; // TODO by default

  if (vm.count("model") && (vm.count("orb") || vm.count("akaze"))) {
    string m = vm["model"].as<string>();
    if (m == "HOMOGRAPHY")
      config.model = FBConfig::HOMOGRAPHY;
    else if (m == "AFFINE")
      config.model = FBConfig::AFFINE;
    else if (m == "RIGID")
      config.model = FBConfig::AFFINE_PARTIAL;
    else {
      cout << "Invalid 'model' argument : choose motion model between HOMOGRAPHY, AFFINE, or RIGID." << endl;
      return -1;
    }
  }

  if (vm.count("model") && vm.count("correlation")) {
    string m = vm["model"].as<string>();
    if (m == "HOMOGRAPHY")
      motion_model = cv::MOTION_HOMOGRAPHY;
    else if (m == "AFFINE")
      motion_model = cv::MOTION_AFFINE;
    else if (m == "RIGID")
      motion_model = cv::MOTION_EUCLIDEAN;
    else if (m == "TRANSLATION")
      motion_model = cv::MOTION_TRANSLATION;
    else {
      cout << "Invalid 'model' argument : choose motion model between HOMOGRAPHY, AFFINE, RIGID or TRANSLATION." << endl;
      return -1;
    }
  }

  if (vm.count("features-matching") && (vm.count("orb") || vm.count("akaze"))) {
    string ft = vm["features-matching"].as<string>();
    if (ft == "RANSAC")
      config.featuresMatching = FBConfig::RANSAC_METHOD;
    else if (ft == "LMEDS")
      config.featuresMatching = FBConfig::LMEDS_METHOD;
    else {
      cout << "Invalid 'features-matching' argument : choose between RANSAC or LMEDS." << endl;
      return -1;
    }
  }

  if (vm.count("correlation")) {
    warp_mat = enhancedCorrelationCoefficientMaximization(ref_img_preprocessed, sensed_img_preprocessed, motion_model);
    warp_img = imgRegistration(ref_img_preprocessed, sensed_img_preprocessed, warp_mat);
  } else if (vm.count("fmt")) {
    warp_mat = fourierMellinTransform(ref_img_preprocessed, sensed_img_preprocessed);
    warp_img = imgRegistration(ref_img_preprocessed, sensed_img_preprocessed, warp_mat);
  } else if (vm.count("orb")) {
    config.detectorDescriptor = FBConfig::ORB_ALGO;
    warp_mat = featuresBasedMethod(ref_img_preprocessed, sensed_img_preprocessed, config);
    warp_img = imgRegistration(ref_img_preprocessed, sensed_img_preprocessed, warp_mat);
    match_img = findMatchFeatures(ref_img_preprocessed, sensed_img_preprocessed).imgOfMatches;
  } else if (vm.count("akaze")) {
    config.detectorDescriptor = FBConfig::AKAZE_ALGO;
    warp_mat = featuresBasedMethod(ref_img_preprocessed, sensed_img_preprocessed, config);
    warp_img = imgRegistration(ref_img_preprocessed, sensed_img_preprocessed, warp_mat);
    match_img = findMatchFeatures(ref_img_preprocessed, sensed_img_preprocessed).imgOfMatches;
  }

  // Show final result
  namedWindow("Reference image", WINDOW_GUI_NORMAL);
  resizeWindow("Reference image", 300, 300);
  moveWindow("Reference image", 100, 100);
  imshow("Reference image", ref_img);

  namedWindow("Sensed image", WINDOW_GUI_NORMAL);
  resizeWindow("Sensed image", 300, 300);
  moveWindow("Sensed image", 500, 100);
  imshow("Sensed image", sensed_img);

  namedWindow("Registered image", WINDOW_GUI_NORMAL);
  resizeWindow("Registered image", 300, 300);
  moveWindow("Registered image", 900, 100);
  imshow("Registered image", warp_img);

  if (!match_img.empty()) {
    namedWindow("Matches", WINDOW_GUI_NORMAL);
    resizeWindow("Matches", 1200, 500);
    moveWindow("Matches", 100, 500);
    imshow("Matches", match_img);
  }

  createButton("button", callbackButton, NULL, QT_PUSH_BUTTON, 0);

  waitKey(0);
}
