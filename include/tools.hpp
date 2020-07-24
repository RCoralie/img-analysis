#ifndef TOOLS_H
#define TOOLS_H
#include <opencv2/core.hpp>
/**
 * @brief Apply erosion on an image
 *
 * @param p_src the source image
 * @param p_dst the result image
 * @param p_erosionType the erosion type (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
 * @param p_erosionSize the erosion size
 */
void erosion(cv::Mat p_src, cv::Mat &p_dst, int p_erosionType, int p_erosionSize);

/**
 * @brief Apply dilatation on an image
 *
 * @param p_src the source image
 * @param p_dst the result image
 * @param p_dilationType the dilatation type (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
 * @param p_dilationSize the dilatation size
 */
void dilation(cv::Mat p_src, cv::Mat &p_dst, int p_dilationType, int p_dilationSize);

/**
 * @brief Draw lines on an image
 *
 * @param p_img the image on which drawing
 * @param p_lines the lines to draw
 */
void drawLines(cv::Mat &p_img, std::vector<cv::Vec4i> p_lines);

/**
 * @brief Draw lines on an image
 *
 * @param p_img the image on which drawing
 * @param p_lines the lines to draw
 */
void drawLines(cv::Mat &p_img, std::vector<cv::Vec2f> p_lines);

/**
 * @brief Apply an hanning window on an image
 *
 * @param p_img the matrix to compute
 */
void hann(cv::Mat &p_img);

/**
 * @brief Compute local dispersion on a 1D signal
 * @details Allows to remove the noise on a 1D signal represented by a 1D matrix
 * and highlights significant local changes
 *
 * @param p_sig1D the one dimension source matrix (source signal)
 * @param p_result the one dimension result matrix (result signal)
 */
void localDispersion1D(cv::Mat p_sig1D, cv::Mat &p_result);

/**
 * @brief Compute peak on a 1D signal
 * @details Allows to conserve only peaks on a 1D signal represented by a 1D matrix,
 * everything else is considered noise and is reduced to zero amplitude
 *
 * @param p_sig1D the one dimension source matrix (source signal)
 * @param p_result the one dimension result matrix (result signal)
 */
void findPeaksIn1DSignal(cv::Mat p_sig1D, cv::Mat &p_result);

#endif
