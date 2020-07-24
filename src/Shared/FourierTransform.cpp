#include "fouriertransform.hpp"

using namespace cv;
using namespace std;

//----------------------------------------------------------------------------//
void IDFT(Mat p_complex, Mat &p_result) { idft(p_complex, p_result, DFT_SCALE | DFT_REAL_OUTPUT); }

//----------------------------------------------------------------------------//
void IDFT(Mat p_magnitude, Mat p_phase, Mat &p_result) {
  Mat dft_complex;
  Mat planes[2];
  polarToCart(p_magnitude, p_phase, planes[0], planes[1]);
  merge(planes, 2, dft_complex);
  fftShift(dft_complex);
  idft(dft_complex, p_result, DFT_REAL_OUTPUT);
}

//----------------------------------------------------------------------------//
void DFT(Mat p_source, Mat &p_complex) {
  p_source.convertTo(p_complex, CV_32F);
  // expand input image to optimal size : on the border add zero values
  Mat padded;
  int m = getOptimalDFTSize(p_source.rows);
  int n = getOptimalDFTSize(p_source.cols); // on the border add zero values
  copyMakeBorder(p_source, padded, 0, m - p_source.rows, 0, n - p_source.cols, BORDER_CONSTANT, Scalar::all(0));
  Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
  merge(planes, 2, p_complex);                   // Add to the expanded another plane with zeros
  dft(p_complex, p_complex, DFT_COMPLEX_OUTPUT); // this way the result may fit in the source matrix
}

//----------------------------------------------------------------------------//
void DFT(Mat p_source, Mat &p_magnitude, Mat &p_phase) {
  Mat fI1;
  p_source.convertTo(fI1, CV_32F);

  // expand input image to optimal size
  int m = getOptimalDFTSize(p_source.rows);
  int n = getOptimalDFTSize(p_source.cols);

  Mat padded1;

  // on the border add zero values
  copyMakeBorder(fI1, padded1, 0, m - p_source.rows, 0, n - p_source.cols, BORDER_CONSTANT, Scalar::all(0));

  // Perform DFT
  Mat fourierTransform1;
  Mat planes1[2];
  dft(fI1, fourierTransform1, DFT_SCALE | DFT_COMPLEX_OUTPUT);
  fftShift(fourierTransform1);
  split(fourierTransform1, planes1); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

  p_magnitude.zeros(planes1[0].rows, planes1[0].cols, CV_32F);
  p_phase.zeros(planes1[0].rows, planes1[0].cols, CV_32F);
  cartToPolar(planes1[0], planes1[1], p_magnitude, p_phase);
}

//----------------------------------------------------------------------------//
void fftShift(Mat &p_matToShift) {
  // crop if it has an odd number of rows or columns
  p_matToShift = p_matToShift(Rect(0, 0, p_matToShift.cols & -2, p_matToShift.rows & -2));

  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = p_matToShift.cols / 2;
  int cy = p_matToShift.rows / 2;

  Mat q0(p_matToShift, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  Mat q1(p_matToShift, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(p_matToShift, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(p_matToShift, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp;
  // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  // swap quadrant (Top-Right with Bottom-Left)
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

//----------------------------------------------------------------------------//
void magnitudeSpectrum(Mat p_complex, Mat &p_magnitude) {
  // Split the complex (2 channel) image in two separate images containing the real and imaginary part
  Mat planes[] = {Mat::zeros(p_complex.size(), CV_32F), Mat::zeros(p_complex.size(), CV_32F)};
  split(p_complex, planes);

  // Use the cartToPolar function to get the phase (angle) and magnitude : log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  // cartToPolar(planes[0], planes[1], planes[0], planes[1]);
  // p_magnitude = planes[0];  // planes[0] = Re(DFT(I) = magnitude = sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  magnitude(planes[0], planes[1], p_magnitude); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)

  // switch to logarithmic scale: log(1 + magnitude)
  p_magnitude += Scalar::all(1);
  log(p_magnitude, p_magnitude);

  // rearrage quadrants
  fftShift(p_magnitude);

  // Transform the magnitude matrix into a viewable image (float values 0-1)
  normalize(p_magnitude, p_magnitude, 1, 0, NORM_INF);
}

//----------------------------------------------------------------------------//
void phaseSpectrum(Mat p_complex, Mat &p_phase) {
  // Split the complex (2 channel) image in two separate images containing the real and imaginary part
  Mat planes[] = {Mat::zeros(p_complex.size(), CV_32F), Mat::zeros(p_complex.size(), CV_32F)};
  split(p_complex, planes);

  // Use the cartToPolar function to get the phase (angle) and magnitude : log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  // cartToPolar(planes[0], planes[1], planes[0], planes[1]);
  // p_phase = planes[1];  // planes[1] = Im(DFT(I)) = phase (the angles are measured in radians (from 0 to 2*Pi))
  phase(planes[0], planes[1], p_phase, false);

  // rearrage quadrants
  fftShift(p_phase);

  // Transform the magnitude matrix into a viewable image (float values 0-1)
  normalize(p_phase, p_phase, 1, 0, NORM_INF);
}
