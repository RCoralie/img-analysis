#include "fouriertransform.hpp"
#include "tools.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QStandardItemModel>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidget>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace transform_mat;

int size = 400;
QLabel *ifft_pixmap;
QLabel *phase_pixmap;
QLabel *magnitude_pixmap;
QLabel *input_img_pixmap;

//----------------------------------------------------------------------------//
cv::Mat transform(const cv::Mat &img, int x, int y, int r, float s) {
  cv::Mat result;

  cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, x, 0, 1, y);
  cv::Mat scale_and_rotation_mat = getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), r, s);

  cv::Mat global_mat = combineAffines(translation_mat, scale_and_rotation_mat);
  cv::warpAffine(img, result, global_mat, img.size());

  return result;
}

//----------------------------------------------------------------------------//
void process(const cv::Mat &img, int x, int y, int r, float s) {

  // Apply transformation
  cv::Mat img_mod = transform(img, x, y, r, s);

  // Apply the gaussian blur
  cv::Mat blured_img;
  GaussianBlur(img_mod, blured_img, Size(29, 29), 0, 0, BORDER_DEFAULT);

  // Hanning window
  cv::Mat hanning_img;
  blured_img.convertTo(hanning_img, CV_64FC1);
  hann(hanning_img);

  // Compute FFT
  cv::Mat fft;
  DFT(hanning_img, fft);
  Mat fft_mag;
  magnitudeSpectrum(fft, fft_mag);
  Mat fft_phase;
  phaseSpectrum(fft, fft_phase);

  // Compute IFFT
  Mat inverse_fft;
  IDFT(fft, inverse_fft);

  // Display results
  cv::Mat input;
  hanning_img.convertTo(input, CV_8U);
  input_img_pixmap->setPixmap(
      QPixmap::fromImage(QImage(input.data, input.cols, input.rows, input.step, QImage::Format_Grayscale8)).scaled(size, size, Qt::KeepAspectRatio));

  cv::Mat mag;
  fft_mag.convertTo(mag, CV_8U, 255);
  magnitude_pixmap->setPixmap(
      QPixmap::fromImage(QImage(mag.data, mag.cols, mag.rows, mag.step, QImage::Format_Grayscale8)).scaled(size, size, Qt::KeepAspectRatio));

  cv::Mat phase;
  fft_phase.convertTo(phase, CV_8U, 255);
  phase_pixmap->setPixmap(
      QPixmap::fromImage(QImage(phase.data, phase.cols, phase.rows, phase.step, QImage::Format_Grayscale8)).scaled(size, size, Qt::KeepAspectRatio));

  cv::Mat ifft;
  inverse_fft.convertTo(ifft, CV_8U);
  ifft_pixmap->setPixmap(
      QPixmap::fromImage(QImage(ifft.data, ifft.cols, ifft.rows, ifft.step, QImage::Format_Grayscale8)).scaled(size, size, Qt::KeepAspectRatio));
}

//----------------------------------------------------------------------------//
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "need one image." << std::endl;
    return -1;
  }

  cv::Mat img = imread(argv[1], IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  //----------
  QApplication application(argc, argv);
  QMainWindow main_window;

  QWidget *main_widget = new QWidget();
  QGridLayout *main_layout = new QGridLayout();
  main_layout->setSizeConstraint(QLayout::SetFixedSize);
  main_widget->setLayout(main_layout);
  main_window.setCentralWidget(main_widget);

  //---------TOOLBAR
  int tr_x = 0;
  int tr_y = 0;
  int rotation = 0;
  double scale = 1.0;

  QToolBar *toolbar = main_window.addToolBar("toolbar");

  QSpinBox *translation_x_box = new QSpinBox(toolbar);
  translation_x_box->setRange(-100, 100);
  translation_x_box->setPrefix("x:");
  translation_x_box->setSingleStep(1);
  translation_x_box->setValue(tr_x);
  toolbar->addWidget(translation_x_box);
  QObject::connect(translation_x_box, QOverload<int>::of(&QSpinBox::valueChanged), [&](int value) {
    tr_x = value;
    process(img, tr_x, tr_y, rotation, scale);
  });

  QSpinBox *translation_y_box = new QSpinBox(toolbar);
  translation_y_box->setRange(-100, 100);
  translation_y_box->setPrefix("y:");
  translation_y_box->setSingleStep(1);
  translation_y_box->setValue(tr_y);
  toolbar->addWidget(translation_y_box);
  QObject::connect(translation_y_box, QOverload<int>::of(&QSpinBox::valueChanged), [&](int value) {
    tr_y = value;
    process(img, tr_x, tr_y, rotation, scale);
  });

  QSpinBox *rotation_box = new QSpinBox(toolbar);
  rotation_box->setRange(-180, 180);
  rotation_box->setPrefix("rotation:");
  rotation_box->setSingleStep(1);
  rotation_box->setValue(rotation);
  toolbar->addWidget(rotation_box);
  QObject::connect(rotation_box, QOverload<int>::of(&QSpinBox::valueChanged), [&](int value) {
    rotation = value;
    process(img, tr_x, tr_y, rotation, scale);
  });

  QDoubleSpinBox *scale_box = new QDoubleSpinBox(toolbar);
  scale_box->setRange(0, 1);
  scale_box->setPrefix("rotation:");
  scale_box->setSingleStep(0.05);
  scale_box->setValue(scale);
  toolbar->addWidget(scale_box);
  QObject::connect(scale_box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double value) {
    scale = value;
    process(img, tr_x, tr_y, rotation, scale);
  });

  //----------DISPLAY

  QWidget *input_img_widget = new QWidget(main_widget);
  QVBoxLayout *input_img_layout = new QVBoxLayout();
  input_img_widget->setLayout(input_img_layout);
  QLabel *input_img_label = new QLabel(input_img_widget);
  input_img_pixmap = new QLabel(input_img_widget);
  input_img_label->setText("Input image");
  input_img_layout->addWidget(input_img_label);
  input_img_layout->addWidget(input_img_pixmap);
  main_layout->addWidget(input_img_widget, 0, 0);

  QWidget *magnitude_widget = new QWidget(main_widget);
  QVBoxLayout *magnitude_layout = new QVBoxLayout();
  magnitude_widget->setLayout(magnitude_layout);
  QLabel *magnitude_label = new QLabel(magnitude_widget);
  magnitude_pixmap = new QLabel(magnitude_widget);
  magnitude_label->setText("Magnitude spectrum");
  magnitude_layout->addWidget(magnitude_label);
  magnitude_layout->addWidget(magnitude_pixmap);
  main_layout->addWidget(magnitude_widget, 0, 1);

  QWidget *phase_widget = new QWidget(main_widget);
  QVBoxLayout *phase_layout = new QVBoxLayout();
  phase_widget->setLayout(phase_layout);
  QLabel *phase_label = new QLabel(phase_widget);
  phase_pixmap = new QLabel(phase_widget);
  phase_label->setText("Phase spectrum");
  phase_layout->addWidget(phase_label);
  phase_layout->addWidget(phase_pixmap);
  main_layout->addWidget(phase_widget, 0, 2);

  QWidget *ifft_widget = new QWidget(main_widget);
  QVBoxLayout *ifft_layout = new QVBoxLayout();
  ifft_widget->setLayout(ifft_layout);
  QLabel *ifft_label = new QLabel(ifft_widget);
  ifft_pixmap = new QLabel(ifft_widget);
  ifft_label->setText("Inverse Fourier transform");
  ifft_layout->addWidget(ifft_label);
  ifft_layout->addWidget(ifft_pixmap);
  main_layout->addWidget(ifft_widget, 0, 3);

  main_window.show();
  process(img, tr_x, tr_y, rotation, scale);

  return application.exec();
}
