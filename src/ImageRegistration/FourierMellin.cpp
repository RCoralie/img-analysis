/**
 * Fourier-Mellin Transformation:
 *  - 1. Discrete Fourier Transform (DFT) to convert images into frequency domain.
 *  - 2. Smoothing and hight-pass filter to avoid the "plus" artifact caused by borders and aliasing artifacts due to rotation.
 *  - 3. Log-polar transform to convert rotation and scaling in the Cartesian coordinate system to translations in the log-polar coordinate system.
 *  - 4. First phase correlation to estimate the rotation and scale difference between the two input images.
 *  - 5. The sensed image is then rotated and scaled to match the second image.
 *  - 6. Second phase correlation to find the translational offset between previous computed image and ref image.
 *
 * It is necessary to first de-rotate and de-scale the image before finding the translational offset.
 * This is because the recovery of translation is not invariant to rotation and scale, whereas the recovery of rotation and scale is invariant to
 * translation. It's the aim of the first phase correlation, done done after first representing the rotation and scaling as translations using the
 * log-polar transform of the magnitude of the Fourier transforms of the images.
 */

#include "opencv2/opencv.hpp"
#include "registration.hpp"

using namespace std;
using namespace cv;

namespace registration {

  namespace fmt {

    namespace {

      /**
       * Apodization, being defined as the multiplication of a sinc function by some window, corresponds in the Fourier domain to a convolution-based
       * construction. The Hanning window is one out of several attempts to design a window that has favorable properties in the Fourier domain.
       *
       * @param src - Source image
       * @param dst - Filtered image
       */
      void hanningApodization(const Mat &src, Mat &dst) {
        CV_Assert(src.type() == CV_32FC1 || src.type() == CV_64FC1);

        Mat hann_window;
        createHanningWindow(hann_window, src.size(), src.type());
        dst = src.mul(hann_window);
      }

      /**
       * Calcul the Discret Fourier Transform on input image and compute the associated fourier spectrum.
       *
       * @param src - Source image
       * @param dst - Dest spectrum of the image
       */
      void fourier(const Mat &src, Mat &dst) {
        Mat padded; // expand input image to optimal size
        int m = getOptimalDFTSize(src.rows);
        int n = getOptimalDFTSize(src.cols); // on the border add zero values
        copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

        Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
        Mat complexI;
        merge(planes, 2, complexI); // Add to the expanded another plane with zeros

        dft(complexI, complexI); // this way the result may fit in the source matrix

        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

        magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
        dst = planes[0];

        dst += Scalar::all(1); // switch to logarithmic scale
        log(dst, dst);

        // crop the spectrum, if it has an odd number of rows or columns
        dst = dst(Rect(0, 0, dst.cols & -2, dst.rows & -2));

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = dst.cols / 2;
        int cy = dst.rows / 2;

        Mat q0(dst, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1(dst, Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(dst, Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(dst, Rect(cx, cy, cx, cy)); // Bottom-Right

        Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);

        normalize(dst, dst, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).
      }

      /**
       * Create an highpass filter
       * @param size  - size of the filter matrix
       * @param dst   - computed filter matrix
       */
      void highpass(Size size, Mat &dst) {
        Mat a = Mat(size.height, 1, CV_32FC1);
        Mat b = Mat(1, size.width, CV_32FC1);

        float step_y = CV_PI / size.height;
        float val = -CV_PI * 0.5;

        for (int i = 0; i < size.height; ++i) {
          a.at<float>(i) = cos(val);
          val += step_y;
        }

        val = -CV_PI * 0.5;
        float step_x = CV_PI / size.width;
        for (int i = 0; i < size.width; ++i) {
          b.at<float>(i) = cos(val);
          val += step_x;
        }

        Mat tmp = a * b;
        dst = (1.0 - tmp).mul(2.0 - tmp);
      }

    } // namespace

    //----------------------------------------------------------------------------//
    Mat fourierMellinTransform(const Mat &ref_img, const Mat &sensed_img) {
      CV_Assert(ref_img.cols != 0 && ref_img.rows != 0 && sensed_img.cols != 0 && sensed_img.rows != 0);
      CV_Assert(ref_img.size() == sensed_img.size());

      Mat block_a = ref_img.clone();
      Mat block_b = sensed_img.clone();

      if (block_a.channels() == 3) {
        cvtColor(block_a, block_a, COLOR_BGR2GRAY);
      }
      if (block_b.channels() == 3) {
        cvtColor(block_b, block_b, COLOR_BGR2GRAY);
      }
      if (block_a.type() == CV_8UC1) {
        block_a.convertTo(block_a, CV_64FC1, 1.0 / 255.0);
      }
      if (block_b.type() == CV_8UC1) {
        block_b.convertTo(block_b, CV_64FC1, 1.0 / 255.0);
      }
      if (block_a.type() == CV_32FC1) {
        block_a.convertTo(block_a, CV_64FC1);
      }
      if (block_b.type() == CV_32FC1) {
        block_b.convertTo(block_b, CV_64FC1);
      }

      if (block_a.channels() == 3) {
        cvtColor(block_a, block_a, COLOR_BGR2GRAY);
      }
      if (block_b.channels() == 3) {
        cvtColor(block_b, block_b, COLOR_BGR2GRAY);
      }
      if (block_a.type() == CV_8UC1) {
        block_a.convertTo(block_a, CV_64FC1, 1.0 / 255.0);
      }
      if (block_b.type() == CV_8UC1) {
        block_b.convertTo(block_b, CV_64FC1, 1.0 / 255.0);
      }
      if (block_a.type() == CV_32FC1) {
        block_a.convertTo(block_a, CV_64FC1);
      }
      if (block_b.type() == CV_32FC1) {
        block_b.convertTo(block_b, CV_64FC1);
      }

      CV_Assert(block_a.type() == CV_32FC1 || block_a.type() == CV_64FC1);
      CV_Assert(block_b.type() == CV_32FC1 || block_b.type() == CV_64FC1);

      // Apply hanning windows to avoid the “plus” artifact caused by borders
      Mat hann_a, hann_b;
      hanningApodization(block_a, hann_a);
      hanningApodization(block_a, hann_b);

      // 1- Discrete Fourier Transform (DFT) to convert images into frequency domain
      Mat ft_a, ft_b;
      fourier(hann_a, ft_a);
      fourier(hann_b, ft_b);

      // 2- Apply highpass filter on spectrums
      Mat highpass_filter;
      highpass(ft_a.size(), highpass_filter);
      ft_a = ft_a.mul(highpass_filter);
      ft_b = ft_b.mul(highpass_filter);

      // 3- Log-polar transform to convert rotation and scaling in the Cartesian coordinate system to translations in the log-polar coordinate system
      Mat lp_a, lp_b;
      Point2f center(ft_a.size() / 2);
      double radius = (double)ft_a.cols / 4;
      double M = (double)ft_a.cols / log(radius);
      logPolar(ft_a, lp_a, center, M, INTER_LINEAR + WARP_FILL_OUTLIERS);
      logPolar(ft_b, lp_b, center, M, INTER_LINEAR + WARP_FILL_OUTLIERS);

      // 4- Log-polar phase correlation to estimate the rotation and scale difference between the two input images.
      Point2f shift = phaseCorrelate(lp_b, lp_a);

      // 5- Rotate and scale the first image to match the second image. This results in block_b_rs.

      /// compute angle
      double angle = -shift.y * 360.0 / block_a.rows;
      if (std::abs(angle) > 90)
        angle += 180.0;
      /// compute scale factor
      double scale = 1.0 / exp(shift.x / M);
      Mat transform_mat = getRotationMatrix2D(center, angle, scale);
      /// rotate and scale the sensed image
      Mat block_b_rs;
      warpAffine(block_b, block_b_rs, transform_mat, block_b.size());

      // 6- final phase correlation used to find the translational offset between rotate and scale sensed image and the reference image.
      Point2f transl = phaseCorrelate(block_a, block_b_rs);

      // Create and return the global transformationn matrice
      double tmp_trans_mat[2][3] = {1, 0, -transl.x, 0, 1, -transl.y}; //[1,0,-x],[0,1,-y]]
      Mat transl_mat(2, 3, CV_64F);
      std::memcpy(transl_mat.data, tmp_trans_mat, 2 * 3 * sizeof(double));
      return combineAffines(transform_mat, transl_mat);
    }

  } // namespace fmt

} // namespace registration
