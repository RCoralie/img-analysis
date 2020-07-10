#include "iostream"
#include "opencv2/opencv.hpp"
#include "registration.hpp"
#include "time.h"

using namespace std;
using namespace cv;
using namespace registration;
using namespace registration::fmt;
using namespace registration::featuresbased;
using namespace registration::corr;

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
  Mat corr_euclidean_reg;
  warpAffine(sensed_img, corr_euclidean_reg, corr_euclidean, ref_img.size());
  namedWindow("ECC (EUCLIDEAN MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ECC (EUCLIDEAN MODEL)", 300, 300);
  moveWindow("ECC (EUCLIDEAN MODEL)", 350, 10);
  imshow("ECC (EUCLIDEAN MODEL)", corr_euclidean_reg);
  cout << "ECC (EUCLIDEAN MODEL): -----------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_euclidean) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  t0 = clock();
  Mat corr_translation = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_TRANSLATION);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  Mat corr_translation_reg;
  warpAffine(sensed_img, corr_translation_reg, corr_translation, ref_img.size());
  namedWindow("ECC (TRANSLATION MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ECC (TRANSLATION MODEL)", 300, 300);
  moveWindow("ECC (TRANSLATION MODEL)", 350, 400);
  imshow("ECC (TRANSLATION MODEL)", corr_translation_reg);
  cout << "ECC (TRANSLATION MODEL): ---------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(corr_translation) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  waitKey(0);

  // Features features methods
  FBConfig config_orb_affine;
  config_orb_affine.model = FBConfig::AFFINE;
  config_orb_affine.detectorDescriptor = FBConfig::ORB_ALGO;
  t0 = clock();
  Mat orb_affine = featuresBasedMethod(ref_img, sensed_img, config_orb_affine);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  Mat orb_affine_reg;
  warpAffine(sensed_img, orb_affine_reg, orb_affine, ref_img.size());
  namedWindow("ORB (AFFINE MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ORB (AFFINE MODEL)", 300, 300);
  moveWindow("ORB (AFFINE MODEL)", 650, 10);
  imshow("ORB (AFFINE MODEL)", orb_affine_reg);
  cout << "ORB (AFFINE MODEL): --------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(orb_affine) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  FBConfig config_orb_rigid;
  config_orb_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_orb_rigid.detectorDescriptor = FBConfig::ORB_ALGO;
  t0 = clock();
  Mat orb_rigid = featuresBasedMethod(ref_img, sensed_img, config_orb_rigid);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  Mat orb_rigid_reg;
  warpAffine(sensed_img, orb_rigid_reg, orb_rigid, ref_img.size());
  namedWindow("ORB (RIGID MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("ORB (RIGID MODEL)", 300, 300);
  moveWindow("ORB (RIGID MODEL)", 650, 400);
  imshow("ORB (RIGID MODEL)", orb_rigid_reg);
  cout << "ORB (RIGID MODEL): ---------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(orb_rigid) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  waitKey(0);

  FBConfig config_akaze_affine;
  config_akaze_affine.model = FBConfig::AFFINE;
  config_akaze_affine.detectorDescriptor = FBConfig::AKAZE_ALGO;
  t0 = clock();
  Mat akaze_affine = featuresBasedMethod(ref_img, sensed_img, config_akaze_affine);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  Mat akaze_affine_reg;
  warpAffine(sensed_img, akaze_affine_reg, akaze_affine, ref_img.size());
  namedWindow("AKAZE (AFFINE MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("AKAZE (AFFINE MODEL)", 300, 300);
  moveWindow("AKAZE (AFFINE MODEL)", 950, 10);
  imshow("AKAZE (AFFINE MODEL)", akaze_affine_reg);
  cout << "AKAZE (AFFINE MODEL): ------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(akaze_affine) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  FBConfig config_akaze_rigid;
  config_akaze_rigid.model = FBConfig::AFFINE_PARTIAL;
  config_akaze_rigid.detectorDescriptor = FBConfig::AKAZE_ALGO;
  t0 = clock();
  Mat akaze_rigid = featuresBasedMethod(ref_img, sensed_img, config_akaze_rigid);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  Mat akaze_rigid_reg;
  warpAffine(sensed_img, akaze_rigid_reg, akaze_rigid, ref_img.size());
  namedWindow("AKAZE (RIGID MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("AKAZE (RIGID MODEL)", 300, 300);
  moveWindow("AKAZE (RIGID MODEL)", 950, 400);
  imshow("AKAZE (RIGID MODEL)", akaze_rigid_reg);
  cout << "AKAZE (RIGID MODEL): -------------------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(akaze_rigid) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  waitKey(0);

  // Fourier-mellin transformation
  t0 = clock();
  Mat fmt = fourierMellinTransform(ref_img, sensed_img);
  duration = (clock() - t0) / (double)CLOCKS_PER_SEC;
  Mat fmt_reg;
  warpAffine(sensed_img, fmt_reg, fmt, ref_img.size());
  namedWindow("FOURIER-MELLIN (AFFINE MODEL)", WINDOW_GUI_NORMAL);
  resizeWindow("FOURIER-MELLIN (AFFINE MODEL)", 300, 300);
  moveWindow("FOURIER-MELLIN (AFFINE MODEL)", 1250, 10);
  imshow("FOURIER-MELLIN (AFFINE MODEL)", fmt_reg);
  cout << "FOURIER-MELLIN (AFFINE MODEL): ---------------------------" << endl;
  cout << "t[x, y] : " << extractTranslationFromAffine(fmt) << endl;
  cout << "execution time : " << duration << " secondes" << endl;

  waitKey(0);
}

//----------------------------------------------------------------------------//
int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "need a reference image." << endl;
    return -1;
  }

  // Read reference image
  Mat ref_img = imread(argv[1]);
  if (ref_img.empty()) {
    cout << "Could not open or find the reference image" << endl;
    return -1;
  }

  float x = 10;
  float y = 20;
  // Create image to be aligned
  Mat sensed_img = ref_img.clone();
  cv::Point2f t(x, y);
  shift(sensed_img, t.x, t.y);

  cout << "GROUND STATE : -------------------------------------------" << endl;
  cout << "t[x, y] : " << t << endl;

  compareShiftRegistrationAccuracy(ref_img, sensed_img);

  compareShiftRegistrationSpeed(ref_img, sensed_img);

  // // Warp
  // Mat imReg;
  // warpAffine(sensed_img, imReg, transform, ref_img.size());
  //
  // // Display results
  // namedWindow("Reference image", WINDOW_GUI_NORMAL);
  // resizeWindow("Reference image", 300, 300);
  // moveWindow("Reference image", 0, 10);
  // imshow("Reference image", ref_img);
  //
  // namedWindow("Sensed image", WINDOW_GUI_NORMAL);
  // resizeWindow("Sensed image", 300, 300);
  // moveWindow("Sensed image", 400, 10);
  // imshow("Sensed image", sensed_img);
  //
  // namedWindow("Registered image", WINDOW_GUI_NORMAL);
  // resizeWindow("Registered image", 300, 300);
  // moveWindow("Registered image", 800, 10);
  // imshow("Registered image", imReg);
  //
  // waitKey(0);
}
