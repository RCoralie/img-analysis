#include "edgedetect.hpp"
#include "registration.hpp"
#include <QApplication>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QStandardItemModel>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidget>
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

QWidget *warp;
QWidget *match;
QLabel *warp_img_pixmap;
QLabel *match_img_pixmap;
QComboBox *model;
QComboBox *features_matching;

//------------------------------------------------------------------------------
void process(const QString &method, const QString &model, const QString &features_matching, Mat &ref_img, Mat &sensed_img) {

  // TODO : OPTIONAL PREPROCESS
  // Mat ref_img_preprocessed = ref_img.clone();
  // Mat sensed_img_preprocessed = sensed_img.clone();
  //   cv::cvtColor(ref_img, ref_img_preprocessed, cv::COLOR_RGB2GRAY);
  //   cv::cvtColor(sensed_img, sensed_img_preprocessed, cv::COLOR_RGB2GRAY);
  //   DericheGradient(ref_img_preprocessed, ref_img_preprocessed, 0.5);
  //   DericheGradient(sensed_img_preprocessed, sensed_img_preprocessed, 0.5);

  std::cout << "Process registration ..." << std::endl;

  Mat warp_mat, warp_img, match_img;
  FBConfig config;
  int motion_model;

  if (model == QString("TRANSLATION")) {
    motion_model = cv::MOTION_TRANSLATION;
  } else if (model == QString("RIGID")) {
    config.model = FBConfig::AFFINE_PARTIAL;
    motion_model = cv::MOTION_EUCLIDEAN;
  } else if (model == QString("AFFINE")) {
    config.model = FBConfig::AFFINE;
    motion_model = cv::MOTION_AFFINE;
  } else {
    config.model = FBConfig::HOMOGRAPHY;
    motion_model = cv::MOTION_HOMOGRAPHY;
  }

  if (features_matching == QString("LMEDS"))
    config.featuresMatching = FBConfig::LMEDS_METHOD;
  else {
    config.featuresMatching = FBConfig::RANSAC_METHOD;
  }

  if (method == QString("Correlation")) {
    warp_mat = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, motion_model);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
  } else if (method == QString("Fourier-Mellin")) {
    warp_mat = fourierMellinTransform(ref_img, sensed_img);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
  } else if (method == QString("ORB")) {
    config.detectorDescriptor = FBConfig::ORB_ALGO;
    warp_mat = featuresBasedMethod(ref_img, sensed_img, config);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
    match_img = findMatchFeatures(ref_img, sensed_img).imgOfMatches;
  } else if (method == QString("AKAZE")) {
    config.detectorDescriptor = FBConfig::AKAZE_ALGO;
    warp_mat = featuresBasedMethod(ref_img, sensed_img, config);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
    match_img = findMatchFeatures(ref_img, sensed_img).imgOfMatches;
  }
  // cvtColor(warp_img, warp_img, COLOR_BGR2RGB);
  warp_img_pixmap->setPixmap(QPixmap::fromImage(QImage(warp_img.data, warp_img.cols, warp_img.rows, warp_img.step, QImage::Format_RGB888)));

  if (!match_img.empty()) {
    // cvtColor(match_img, match_img, COLOR_BGR2RGB);
    match_img_pixmap->setPixmap(QPixmap::fromImage(QImage(match_img.data, match_img.cols, match_img.rows, match_img.step, QImage::Format_RGB888)));
    match->setVisible(true);
  } else {
    match->setVisible(false);
  }

  std::cout << "done" << std::endl;
}

//------------------------------------------------------------------------------
void SetComboBoxItemEnabled(QComboBox *comboBox, int index, bool enabled) {
  auto *model = qobject_cast<QStandardItemModel *>(comboBox->model());
  assert(model);
  if (!model)
    return;

  auto *item = model->item(index);
  assert(item);
  if (!item)
    return;
  item->setEnabled(enabled);
}

//------------------------------------------------------------------------------
void registrationMethodChanged(const QString &text) {

  if (text == QString("ORB") || text == QString("AKAZE")) {
    model->setCurrentIndex(2);
    SetComboBoxItemEnabled(model, 3, false);
    features_matching->setEnabled(true);

  } else {
    features_matching->setEnabled(false);
    SetComboBoxItemEnabled(model, 3, true);
  }
}

//------------------------------------------------------------------------------
int main(int argc, char **argv) {

  // Declare the supported options.
  options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");

  desc.add_options()("registration,R", value<vector<string>>()->multitoken(), "image registration : [reference image path] [sensed image path]");
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
  if (vm.count("registration")) {
    arg = vm["registration"].as<vector<string>>();
  } else {
    cout << "See --help for more details." << endl;
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

  // ---------------------------------------------------------------------------
  QApplication application(argc, argv);
  QMainWindow mainWindow;

  QWidget *mainWidget = new QWidget();
  QHBoxLayout *mainLayout = new QHBoxLayout();
  mainWidget->setLayout(mainLayout);
  mainWindow.setCentralWidget(mainWidget);

  QToolBar *toolbar = mainWindow.addToolBar("toolbar for image registration");

  QComboBox *algo = new QComboBox(toolbar);
  toolbar->addWidget(algo);
  algo->addItem("Correlation");
  algo->addItem("ORB");
  algo->addItem("AKAZE");
  algo->addItem("Fourier-Mellin");
  algo->setCurrentIndex(1);
  QObject::connect(algo, QOverload<const QString &>::of(&QComboBox::currentTextChanged),
                   [=](const QString &text) { registrationMethodChanged(text); });

  model = new QComboBox(toolbar);
  toolbar->addWidget(model);
  model->addItem("HOMOGRAPHY");
  model->addItem("AFFINE");
  model->addItem("RIGID");
  model->addItem("TRANSLATION");
  model->setCurrentIndex(2);
  SetComboBoxItemEnabled(model, 3, false);

  features_matching = new QComboBox(toolbar);
  toolbar->addWidget(features_matching);
  features_matching->addItem("RANSAC");
  features_matching->addItem("LMEDS");
  features_matching->setCurrentIndex(0);

  QPushButton *start = new QPushButton(toolbar);
  toolbar->addWidget(start);
  start->setText("Process");
  QObject::connect(start, &QPushButton::clicked,
                   [&]() { process(algo->currentText(), model->currentText(), features_matching->currentText(), ref_img, sensed_img); });

  QWidget *reference = new QWidget(mainWidget);
  QVBoxLayout *ref_layout = new QVBoxLayout();
  reference->setLayout(ref_layout);
  QLabel *ref_img_label = new QLabel(reference);
  QLabel *ref_img_pixmap = new QLabel(reference);
  ref_img_label->setText("Reference Image");
  cvtColor(ref_img, ref_img, COLOR_BGR2RGB);
  ref_img_pixmap->setPixmap(QPixmap::fromImage(QImage(ref_img.data, ref_img.cols, ref_img.rows, ref_img.step, QImage::Format_RGB888)));
  ref_layout->addWidget(ref_img_label);
  ref_layout->addWidget(ref_img_pixmap);
  mainLayout->addWidget(reference);

  QWidget *sensed = new QWidget(mainWidget);
  QVBoxLayout *sensed_layout = new QVBoxLayout();
  sensed->setLayout(sensed_layout);
  QLabel *sensed_img_label = new QLabel(sensed);
  QLabel *sensed_img_pixmap = new QLabel(sensed);
  sensed_img_label->setText("Sensed Image");
  cvtColor(sensed_img, sensed_img, COLOR_BGR2RGB);
  sensed_img_pixmap->setPixmap(QPixmap::fromImage(QImage(sensed_img.data, sensed_img.cols, sensed_img.rows, sensed_img.step, QImage::Format_RGB888)));
  sensed_layout->addWidget(sensed_img_label);
  sensed_layout->addWidget(sensed_img_pixmap);
  mainLayout->addWidget(sensed);

  warp = new QWidget(mainWidget);
  QVBoxLayout *warp_layout = new QVBoxLayout();
  warp->setLayout(warp_layout);
  QLabel *warp_img_label = new QLabel(warp);
  warp_img_pixmap = new QLabel(warp);
  warp_img_label->setText("Warp Image");
  warp_layout->addWidget(warp_img_label);
  warp_layout->addWidget(warp_img_pixmap);
  mainLayout->addWidget(warp);

  match = new QWidget();
  QVBoxLayout *match_layout = new QVBoxLayout();
  match->setLayout(match_layout);
  QLabel *match_img_label = new QLabel(match);
  match_img_pixmap = new QLabel(match);
  match_img_label->setText("Match Image");
  match_layout->addWidget(match_img_label);
  match_layout->addWidget(match_img_pixmap);

  process(algo->currentText(), model->currentText(), features_matching->currentText(), ref_img, sensed_img);

  mainWindow.show();

  return application.exec();
}
