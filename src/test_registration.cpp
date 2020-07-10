#include "registration.hpp"
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;
using namespace registration;
using namespace registration::fmt;
using namespace registration::featuresbased;
using namespace registration::corr;
using namespace boost::program_options;

boost::regex number_regex("-?[0-9]+([\\.][0-9]+)?");

//----------------------------------------------------------------------------//
void shift(cv::Mat &img, int x, int y) {
  cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, x, 0, 1, y);
  cv::warpAffine(img, img, M, img.size());
}

//----------------------------------------------------------------------------//
void rotate(cv::Mat &img, int r) {
  cv::Mat M = getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), r, 1);
  cv::warpAffine(img, img, M, img.size());
}

//----------------------------------------------------------------------------//
void rescale(cv::Mat &img, float s) {
  cv::Mat M = getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), 0, s);
  cv::warpAffine(img, img, M, img.size());
}

//----------------------------------------------------------------------------//
void compareShiftRegistrationAccuracy(const Mat &ref_img, const Mat &sensed_img) {

  // enhancedCorrelationCoefficientMaximization
  Mat corr_affine = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_AFFINE);
  cout << "ECC (AFFINE MODEL): --------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_affine) << endl;

  Mat corr_euclidean = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_EUCLIDEAN);
  cout << "ECC (EUCLIDEAN MODEL): -----------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_euclidean) << endl;

  Mat corr_translation = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_TRANSLATION);
  cout << "ECC (TRANSLATION MODEL): ---------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_translation) << endl;

  // Features features methods
  FBConfig config_orb_affine;
  config_orb_affine.model = FBConfig::AFFINE;
  config_orb_affine.detectorDescriptor = FBConfig::ORB_ALGO;
  Mat tr_orb_affine = featuresBasedMethod(ref_img, sensed_img, config_orb_affine);
  cout << "ORB (AFFINE MODEL) : -------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(tr_orb_affine) << endl;

  FBConfig config_orb_rigid;
  config_orb_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_orb_rigid.detectorDescriptor = FBConfig::ORB_ALGO;
  Mat tr_orb_rigid = featuresBasedMethod(ref_img, sensed_img, config_orb_rigid);
  cout << "ORB (RIGID MODEL) : --------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(tr_orb_rigid) << endl;

  FBConfig config_akaze_affine;
  config_akaze_affine.model = FBConfig::AFFINE;
  config_akaze_affine.detectorDescriptor = FBConfig::AKAZE_ALGO;
  Mat tr_akaze_affine = featuresBasedMethod(ref_img, sensed_img, config_akaze_affine);
  cout << "AKAZE (AFFINE MODEL) : -----------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(tr_akaze_affine) << endl;

  FBConfig config_akaze_rigid;
  config_akaze_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_akaze_rigid.detectorDescriptor = FBConfig::AKAZE_ALGO;
  Mat tr_akaze_rigid = featuresBasedMethod(ref_img, sensed_img, config_akaze_rigid);
  cout << "AKAZE (RIGID MODEL) : ------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(tr_akaze_rigid) << endl;

  // Fourier-mellin transformation
  Mat fmt = fourierMellinTransform(ref_img, sensed_img);
  cout << "FOURIER-MELLIN (AFFINE MODEL): ---------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(fmt) << endl;
}

//----------------------------------------------------------------------------//
void compareShiftRegistrationSpeed(const Mat &ref_img, const Mat &sensed_img) {

  int occurrences_nb = 100;
  clock_t t0;
  double duration;

  // enhancedCorrelationCoefficientMaximization
  t0 = clock();
  Mat corr_affine = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_AFFINE);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "ECC (AFFINE MODEL): --------------------------------------" << endl;
  cout << "mean execution time on 1 runs : " << duration << " secondes" << endl;

  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat corr_euclidean = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_EUCLIDEAN);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "ECC (EUCLIDEAN MODEL): -----------------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat corr_translation = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_TRANSLATION);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "ECC (TRANSLATION MODEL): ---------------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

  // Features features methods
  FBConfig config_orb_affine;
  config_orb_affine.model = FBConfig::AFFINE;
  config_orb_affine.detectorDescriptor = FBConfig::ORB_ALGO;
  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat tr_orb_affine = featuresBasedMethod(ref_img, sensed_img, config_orb_affine);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "ORB (AFFINE MODEL) : -------------------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

  FBConfig config_orb_rigid;
  config_orb_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_orb_rigid.detectorDescriptor = FBConfig::ORB_ALGO;
  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat tr_orb_rigid = featuresBasedMethod(ref_img, sensed_img, config_orb_rigid);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "ORB (RIGID MODEL) : --------------------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

  FBConfig config_akaze_affine;
  config_akaze_affine.model = FBConfig::AFFINE;
  config_akaze_affine.detectorDescriptor = FBConfig::AKAZE_ALGO;
  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat tr_akaze_affine = featuresBasedMethod(ref_img, sensed_img, config_akaze_affine);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "AKAZE (AFFINE MODEL) : -----------------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

  FBConfig config_akaze_rigid;
  config_akaze_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_akaze_rigid.detectorDescriptor = FBConfig::AKAZE_ALGO;
  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat tr_akaze_rigid = featuresBasedMethod(ref_img, sensed_img, config_akaze_rigid);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "AKAZE (RIGID MODEL) : ------------------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

  // Fourier-mellin transformation
  t0 = clock();
  for (int i = 0; i <= occurrences_nb; i++) {
    Mat fmt = fourierMellinTransform(ref_img, sensed_img);
  }
  duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
  cout << "FOURIER-MELLIN (AFFINE MODEL): ---------------------------" << endl;
  cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;
}

//----------------------------------------------------------------------------//
void compareShiftRegistrationVisually(const Mat &ref_img, const Mat &sensed_img) {

  // Display reference image
  namedWindow("Reference image", WINDOW_GUI_NORMAL);
  resizeWindow("Reference image", 300, 300);
  moveWindow("Reference image", 0, 10);
  imshow("Reference image", ref_img);

  // Display sensed image
  namedWindow("Sensed image", WINDOW_GUI_NORMAL);
  resizeWindow("Sensed image", 300, 300);
  moveWindow("Sensed image", 0, 400);
  imshow("Sensed image", sensed_img);

  waitKey(0);

  // Display results

  clock_t t0;
  double duration;

  // enhancedCorrelationCoefficientMaximization
  t0 = clock();
  Mat corr_euclidean = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_EUCLIDEAN);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "ECC (EUCLIDEAN MODEL): -----------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_euclidean) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat corr_euclidean_reg = sensed_img.clone();
  shift(corr_euclidean_reg, extractTranslationFromAffine(corr_euclidean).x, extractTranslationFromAffine(corr_euclidean).y);
  // warpAffine(sensed_img, corr_euclidean_reg, corr_euclidean, ref_img.size());
  namedWindow("ECC (EUCLIDEAN MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ECC (EUCLIDEAN MODEL)", 300, 300);
  moveWindow("ECC (EUCLIDEAN MODEL)", 350, 10);
  imshow("ECC (EUCLIDEAN MODEL)", corr_euclidean_reg);

  t0 = clock();
  Mat corr_translation = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_TRANSLATION);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "ECC (TRANSLATION MODEL): ---------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_translation) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat corr_translation_reg = sensed_img.clone();
  shift(corr_euclidean_reg, extractTranslationFromAffine(corr_translation).x, extractTranslationFromAffine(corr_translation).y);
  namedWindow("ECC (TRANSLATION MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ECC (TRANSLATION MODEL)", 300, 300);
  moveWindow("ECC (TRANSLATION MODEL)", 350, 400);
  imshow("ECC (TRANSLATION MODEL)", corr_translation_reg);

  waitKey(0);

  // Features features methods
  FBConfig config_orb_affine;
  config_orb_affine.model = FBConfig::AFFINE;
  config_orb_affine.detectorDescriptor = FBConfig::ORB_ALGO;
  t0 = clock();
  Mat orb_affine = featuresBasedMethod(ref_img, sensed_img, config_orb_affine);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "ORB (AFFINE MODEL): --------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(orb_affine) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat orb_affine_reg = sensed_img.clone();
  shift(orb_affine_reg, extractTranslationFromAffine(orb_affine).x, extractTranslationFromAffine(orb_affine).y);
  namedWindow("ORB (AFFINE MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ORB (AFFINE MODEL)", 300, 300);
  moveWindow("ORB (AFFINE MODEL)", 650, 10);
  imshow("ORB (AFFINE MODEL)", orb_affine_reg);

  FBConfig config_orb_rigid;
  config_orb_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_orb_rigid.detectorDescriptor = FBConfig::ORB_ALGO;
  t0 = clock();
  Mat orb_rigid = featuresBasedMethod(ref_img, sensed_img, config_orb_rigid);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "ORB (RIGID MODEL): ---------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(orb_rigid) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat orb_rigid_reg = sensed_img.clone();
  shift(orb_rigid_reg, extractTranslationFromAffine(orb_rigid).x, extractTranslationFromAffine(orb_rigid).y);
  namedWindow("ORB (RIGID MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ORB (RIGID MODEL)", 300, 300);
  moveWindow("ORB (RIGID MODEL)", 650, 400);
  imshow("ORB (RIGID MODEL)", orb_rigid_reg);

  waitKey(0);

  FBConfig config_akaze_affine;
  config_akaze_affine.model = FBConfig::AFFINE;
  config_akaze_affine.detectorDescriptor = FBConfig::AKAZE_ALGO;
  t0 = clock();
  Mat akaze_affine = featuresBasedMethod(ref_img, sensed_img, config_akaze_affine);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "AKAZE (AFFINE MODEL): ------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(akaze_affine) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat akaze_affine_reg = sensed_img.clone();
  shift(akaze_affine_reg, extractTranslationFromAffine(akaze_affine).x, extractTranslationFromAffine(akaze_affine).y);
  namedWindow("AKAZE (AFFINE MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("AKAZE (AFFINE MODEL)", 300, 300);
  moveWindow("AKAZE (AFFINE MODEL)", 950, 10);
  imshow("AKAZE (AFFINE MODEL)", akaze_affine_reg);

  FBConfig config_akaze_rigid;
  config_akaze_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_akaze_rigid.detectorDescriptor = FBConfig::AKAZE_ALGO;
  t0 = clock();
  Mat akaze_rigid = featuresBasedMethod(ref_img, sensed_img, config_akaze_rigid);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "AKAZE (RIGID MODEL): -------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(akaze_rigid) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat akaze_rigid_reg = sensed_img.clone();
  shift(akaze_rigid_reg, extractTranslationFromAffine(akaze_rigid).x, extractTranslationFromAffine(akaze_rigid).y);
  namedWindow("AKAZE (RIGID MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("AKAZE (RIGID MODEL)", 300, 300);
  moveWindow("AKAZE (RIGID MODEL)", 950, 400);
  imshow("AKAZE (RIGID MODEL)", akaze_rigid_reg);

  waitKey(0);

  // Fourier-mellin transformation
  t0 = clock();
  Mat fmt = fourierMellinTransform(ref_img, sensed_img);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  cout << "FOURIER-MELLIN (AFFINE MODEL): ---------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(fmt) << endl;
  cout << "execution time : " << duration << " secondes" << endl;
  Mat fmt_reg = sensed_img.clone();
  shift(fmt_reg, extractTranslationFromAffine(fmt).x, extractTranslationFromAffine(fmt).y);
  namedWindow("FOURIER-MELLIN (AFFINE MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("FOURIER-MELLIN (AFFINE MODEL)", 300, 300);
  moveWindow("FOURIER-MELLIN (AFFINE MODEL)", 1250, 10);
  imshow("FOURIER-MELLIN (AFFINE MODEL)", fmt_reg);

  waitKey(0);
}

//----------------------------------------------------------------------------//
int main(int argc, char **argv) {

  // Declare the supported options.
  options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");
  desc.add_options()("benchmark,B", value<vector<string>>()->multitoken(),
                     "compare accuracy and execution time of each registration methods on ground state computed : [reference image path] "
                     "[x-translation] [y-translation] ");
  desc.add_options()("vizualize,V", value<vector<string>>()->multitoken(),
                     "compare registration methods on reference and sensed images loaded : [reference image path] [sensed image path]");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  // --------------------------------------------------------------------------
  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }
  // --------------------------------------------------------------------------
  if (vm.count("benchmark")) {
    vector<string> arg = vm["benchmark"].as<vector<string>>();
    if (arg.size() != 3) {
      cout << "Expected 3 arguments to process the benchmark : [reference image path] [x-translation] [y-translation]" << endl;
      return -1;
    }
    boost::cmatch what;
    if (!boost::regex_match(arg[1].c_str(), what, number_regex) || !boost::regex_match(arg[2].c_str(), what, number_regex)) {
      cout << "Invalid argument type for : [x-translation] or [y-translation]" << endl;
      return -1;
    }
    Mat ref_img = imread(arg[0]);
    if (ref_img.empty()) {
      cout << "Could not open or find the [reference image path]" << endl;
      return -1;
    }

    // compare accuracy and execution time of each registration methods on ground state
    double x = atof(arg[1].c_str());
    double y = atof(arg[2].c_str());

    // Create image to be aligned
    Mat sensed_img = ref_img.clone();
    cv::Point2f t(x, y);
    shift(sensed_img, t.x, t.y);

    cout << "GROUND STATE : -------------------------------------------" << endl;
    cout << "t[x, y] : " << t << endl;

    compareShiftRegistrationAccuracy(ref_img, sensed_img);

    compareShiftRegistrationSpeed(ref_img, sensed_img);
  }

  // --------------------------------------------------------------------------
  else if (vm.count("vizualize")) {
    vector<string> arg = vm["vizualize"].as<vector<string>>();
    if (arg.size() != 2) {
      cout << "Expected 2 arguments : [reference image path] [sensed image path]" << endl;
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

    // compare registration methods on reference and sensed images loaded
    compareShiftRegistrationVisually(ref_img, sensed_img);
  }
}
