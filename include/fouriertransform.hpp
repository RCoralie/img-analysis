#ifndef FT_H
#define FT_H
#include <opencv2/core.hpp>

/**
 * @brief Compute inverse discret frourier transform from complex matrix
 * @details You obtain a image matrix as result
 *
 * @param p_complex the source matrix
 * @param p_result the result matrix
 */
void IDFT(cv::Mat p_complex, cv::Mat &p_result);

/**
 * @brief Compute inverse discret frourier transform from magnitude and phase spectrum
 * @details You obtain a image matrix as result
 *
 * @param p_magnitude the magnitude spectrum
 * @param p_phase the phase spectrum
 * @param p_result the result matrix
 */
void IDFT(cv::Mat p_magnitude, cv::Mat p_phase, cv::Mat &p_result);

/**
 * @brief Compute discret frourier transform to obtain complex matrix
 *
 * @param p_source the source matrix
 * @param p_complex the result complex matrix
 */
void DFT(cv::Mat p_source, cv::Mat &p_complex);

/**
 * @brief Compute discret frourier transform to obtain magnitude and phase spectrum
 * @details The magnitude spectrum is represented with logarithmic scale and swap quadrant,
 * the phase spectrum is represented with swap quadrant (see fftShift function)
 *
 * @param p_source the source matrix
 * @param p_magnitude the result magnitude spectrum
 * @param p_phase the result phase spectrum
 */
void DFT(cv::Mat p_source, cv::Mat &p_magnitude, cv::Mat &p_phase);

/**
 * @brief Rearrange the quadrants of Fourier image so that the origin is at the image center
 * @details You have to calculate the magnitude or phase spectrum before applying this function on the result.
 *
 * @param p_matToShift the matrix to swap
 */
void fftShift(cv::Mat &p_matToShift);

/**
 * @brief Compute the magnitude spectrum from the complex values ​​obtained by the fourier transform
 * @details You have to calculate the fourier transform before applying this function on the result.
 * the magnitude spectrum is represented with logarithmic scale and swap quadrant (see fftShift function)
 *
 * @param p_complex the source matrix contained complex values
 * @param p_magnitude the result matrix contained the magnitude spectrum
 */
void magnitudeSpectrum(cv::Mat p_complex, cv::Mat &p_magnitude);

/**
 * @brief Compute the phase spectrum from the complex values ​​obtained by the fourier transform
 * @details You have to calculate the fourier transform before applying this function on the result.
 * the phase spectrum is represented with swap quadrant (see fftShift function)
 *
 * @param p_complex the source matrix contained complex values
 * @param p_phase the result matrix contained the phase spectrum
 */
void phaseSpectrum(cv::Mat p_complex, cv::Mat &p_phase);

#endif
