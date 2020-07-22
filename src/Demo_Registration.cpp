#include "edgedetect.hpp"
#include "registration.hpp"
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

QWidget *match_widget;
QWidget *preprocess_ref_widget;
QWidget *preprocess_sensed_widget;

QLabel *warp_img_pixmap;
QLabel *match_img_pixmap;
QLabel *preprocess_ref_img_pixmap;
QLabel *preprocess_sensed_img_pixmap;

QComboBox *model_box;
QComboBox *matching_method_box;
QPushButton *matches_btn;
QDoubleSpinBox *deriche_gamma;

bool display_matches = false;
int size = 400;

//------------------------------------------------------------------------------
void preprocess(int state, const Mat &ref_img, const Mat &sensed_img, Mat &ref_img_preprocessed, Mat &sensed_img_preprocessed) {
  ref_img_preprocessed = ref_img.clone();
  sensed_img_preprocessed = sensed_img.clone();
  if (state == 0) {
    preprocess_ref_widget->setVisible(false);
    preprocess_sensed_widget->setVisible(false);
    deriche_gamma->setEnabled(false);

    std::cout << state << std::endl;
  } else {
    DericheGradient(ref_img_preprocessed, ref_img_preprocessed, deriche_gamma->value());
    DericheGradient(sensed_img_preprocessed, sensed_img_preprocessed, deriche_gamma->value());
    preprocess_ref_img_pixmap->setPixmap(QPixmap::fromImage(QImage(ref_img_preprocessed.data, ref_img_preprocessed.cols, ref_img_preprocessed.rows,
                                                                   ref_img_preprocessed.step, QImage::Format_RGB888))
                                             .scaled(size, size, Qt::KeepAspectRatio));
    preprocess_sensed_img_pixmap->setPixmap(
        QPixmap::fromImage(QImage(sensed_img_preprocessed.data, sensed_img_preprocessed.cols, sensed_img_preprocessed.rows,
                                  sensed_img_preprocessed.step, QImage::Format_RGB888))
            .scaled(size, size, Qt::KeepAspectRatio));
    preprocess_ref_widget->setVisible(true);
    preprocess_sensed_widget->setVisible(true);
    deriche_gamma->setEnabled(true);
  }
}
//------------------------------------------------------------------------------
void process(const QString &method, const QString &model, const QString &features_matching, const Mat &ref_img, const Mat &sensed_img,
             Mat &ref_img_preprocessed, Mat &sensed_img_preprocessed) {

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
    warp_mat = enhancedCorrelationCoefficientMaximization(ref_img_preprocessed, sensed_img_preprocessed, motion_model);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
  } else if (method == QString("Fourier-Mellin")) {
    warp_mat = fourierMellinTransform(ref_img_preprocessed, sensed_img_preprocessed);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
  } else if (method == QString("ORB")) {
    config.detectorDescriptor = FBConfig::ORB_ALGO;
    warp_mat = featuresBasedMethod(ref_img_preprocessed, sensed_img_preprocessed, config);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
    match_img = findMatchFeatures(ref_img_preprocessed, sensed_img_preprocessed).imgOfMatches;
  } else if (method == QString("AKAZE")) {
    config.detectorDescriptor = FBConfig::AKAZE_ALGO;
    warp_mat = featuresBasedMethod(ref_img_preprocessed, sensed_img_preprocessed, config);
    warp_img = imgRegistration(ref_img, sensed_img, warp_mat);
    match_img = findMatchFeatures(ref_img_preprocessed, sensed_img_preprocessed).imgOfMatches;
  }
  // cvtColor(warp_img, warp_img, COLOR_BGR2RGB);
  warp_img_pixmap->setPixmap(QPixmap::fromImage(QImage(warp_img.data, warp_img.cols, warp_img.rows, warp_img.step, QImage::Format_RGB888))
                                 .scaled(size * 2 + 100, size * 2 + 100, Qt::KeepAspectRatio));

  if (!match_img.empty()) {
    // cvtColor(match_img, match_img, COLOR_BGR2RGB);
    match_img_pixmap->setPixmap(QPixmap::fromImage(QImage(match_img.data, match_img.cols, match_img.rows, match_img.step, QImage::Format_RGB888))
                                    .scaled(size * 4, size * 2, Qt::KeepAspectRatio));
    matches_btn->setEnabled(true);
  } else {
    match_widget->setVisible(false);
    matches_btn->setEnabled(false);
  }

  std::cout << "done" << std::endl;
}

//------------------------------------------------------------------------------
void SetComboBoxItemEnabled(QComboBox *box, int index, bool enabled) {
  auto *model = qobject_cast<QStandardItemModel *>(box->model());
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
    model_box->setCurrentIndex(2);
    SetComboBoxItemEnabled(model_box, 3, false);
    matching_method_box->setEnabled(true);
  } else {
    matching_method_box->setEnabled(false);
    SetComboBoxItemEnabled(model_box, 3, true);
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

  Mat ref_img_preprocessed = ref_img.clone();
  Mat sensed_img_preprocessed = sensed_img.clone();

  // ---------------------------------------------------------------------------
  QApplication application(argc, argv);
  QMainWindow main_window;

  QWidget *main_widget = new QWidget();
  QGridLayout *main_layout = new QGridLayout();
  main_layout->setSizeConstraint(QLayout::SetFixedSize);
  main_widget->setLayout(main_layout);
  main_window.setCentralWidget(main_widget);

  //---------TOOLBAR REGISTRATION

  QToolBar *toolbar_registration = main_window.addToolBar("toolbar for image registration");

  QComboBox *algo = new QComboBox(toolbar_registration);
  toolbar_registration->addWidget(algo);
  algo->addItem("Correlation");
  algo->addItem("ORB");
  algo->addItem("AKAZE");
  algo->addItem("Fourier-Mellin");
  algo->setCurrentIndex(1);
  QObject::connect(algo, QOverload<const QString &>::of(&QComboBox::currentTextChanged),
                   [=](const QString &text) { registrationMethodChanged(text); });

  model_box = new QComboBox(toolbar_registration);
  toolbar_registration->addWidget(model_box);
  model_box->addItem("HOMOGRAPHY");
  model_box->addItem("AFFINE");
  model_box->addItem("RIGID");
  model_box->addItem("TRANSLATION");
  model_box->setCurrentIndex(2);
  SetComboBoxItemEnabled(model_box, 3, false);

  matching_method_box = new QComboBox(toolbar_registration);
  toolbar_registration->addWidget(matching_method_box);
  matching_method_box->addItem("RANSAC");
  matching_method_box->addItem("LMEDS");
  matching_method_box->setCurrentIndex(0);

  toolbar_registration->addSeparator();

  QPushButton *process_btn = new QPushButton(toolbar_registration);
  toolbar_registration->addWidget(process_btn);
  process_btn->setText("Process");
  QObject::connect(process_btn, &QPushButton::clicked, [&]() {
    process(algo->currentText(), model_box->currentText(), matching_method_box->currentText(), ref_img, sensed_img, ref_img_preprocessed,
            sensed_img_preprocessed);
  });

  toolbar_registration->addSeparator();

  matches_btn = new QPushButton(toolbar_registration);
  toolbar_registration->addWidget(matches_btn);
  matches_btn->setText("Display matches");
  QObject::connect(matches_btn, &QPushButton::clicked, [&]() { match_widget->setVisible(true); });

  //----------TOOLBAR PREPROCESS

  QToolBar *toolbar_preprocess = main_window.addToolBar("toolbar for preprocessing");
  main_window.insertToolBarBreak(toolbar_preprocess);

  QCheckBox *preprocess_btn = new QCheckBox(toolbar_preprocess);
  toolbar_preprocess->addWidget(preprocess_btn);
  preprocess_btn->setText("Preprocess");
  QObject::connect(preprocess_btn, &QCheckBox::stateChanged, [&](int state) {
    preprocess(state, ref_img, sensed_img, ref_img_preprocessed, sensed_img_preprocessed);
    main_window.adjustSize();
  });

  deriche_gamma = new QDoubleSpinBox(toolbar_preprocess);
  deriche_gamma->setRange(0, 3);
  deriche_gamma->setPrefix("gamma:");
  deriche_gamma->setSingleStep(0.25);
  deriche_gamma->setValue(0.5);
  deriche_gamma->setEnabled(false);
  toolbar_preprocess->addWidget(deriche_gamma);
  QObject::connect(deriche_gamma, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                   [&](double d) { preprocess(true, ref_img, sensed_img, ref_img_preprocessed, sensed_img_preprocessed); });

  //----------DISPLAY IMG

  QWidget *reference_widget = new QWidget(main_widget);
  QVBoxLayout *ref_layout = new QVBoxLayout();
  reference_widget->setLayout(ref_layout);
  QLabel *ref_img_label = new QLabel(reference_widget);
  QLabel *ref_img_pixmap = new QLabel(reference_widget);
  ref_img_label->setText("Reference Image");
  cvtColor(ref_img, ref_img, COLOR_BGR2RGB);
  ref_img_pixmap->setPixmap(QPixmap::fromImage(QImage(ref_img.data, ref_img.cols, ref_img.rows, ref_img.step, QImage::Format_RGB888))
                                .scaled(size, size, Qt::KeepAspectRatio));
  ref_layout->addWidget(ref_img_label);
  ref_layout->addWidget(ref_img_pixmap);
  main_layout->addWidget(reference_widget, 0, 0);

  QWidget *sensed_widget = new QWidget(main_widget);
  QVBoxLayout *sensed_layout = new QVBoxLayout();
  sensed_widget->setLayout(sensed_layout);
  QLabel *sensed_img_label = new QLabel(sensed_widget);
  QLabel *sensed_img_pixmap = new QLabel(sensed_widget);
  sensed_img_label->setText("Sensed Image");
  cvtColor(sensed_img, sensed_img, COLOR_BGR2RGB);
  sensed_img_pixmap->setPixmap(QPixmap::fromImage(QImage(sensed_img.data, sensed_img.cols, sensed_img.rows, sensed_img.step, QImage::Format_RGB888))
                                   .scaled(size, size, Qt::KeepAspectRatio));
  sensed_layout->addWidget(sensed_img_label);
  sensed_layout->addWidget(sensed_img_pixmap);
  main_layout->addWidget(sensed_widget, 0, 1);

  QWidget *warp_widget = new QWidget(main_widget);
  QVBoxLayout *warp_layout = new QVBoxLayout();
  warp_widget->setLayout(warp_layout);
  QLabel *warp_img_label = new QLabel(warp_widget);
  warp_img_pixmap = new QLabel(warp_widget);
  warp_img_label->setText("Warp Image");
  warp_img_label->setAlignment(Qt::AlignCenter);
  warp_layout->addWidget(warp_img_label);
  warp_layout->addWidget(warp_img_pixmap);
  main_layout->addWidget(warp_widget, 0, 2, 2, 2);

  //----------MATCH IMAGE FOR FEATURES BASED REGISTRATION ONLY

  match_widget = new QWidget();
  QVBoxLayout *match_layout = new QVBoxLayout();
  match_widget->setLayout(match_layout);
  QLabel *match_img_label = new QLabel(match_widget);
  match_img_pixmap = new QLabel(match_widget);
  match_img_label->setText("Match Image");
  match_layout->addWidget(match_img_label);
  match_layout->addWidget(match_img_pixmap);

  //----------PREPROCESSED IMAGE

  preprocess_ref_widget = new QWidget(main_widget);
  QVBoxLayout *preprocess_ref_layout = new QVBoxLayout();
  preprocess_ref_widget->setLayout(preprocess_ref_layout);
  QLabel *preprocess_ref_img_label = new QLabel(preprocess_ref_widget);
  preprocess_ref_img_pixmap = new QLabel(preprocess_ref_widget);
  preprocess_ref_img_label->setText("Preprocessed Ref Image");
  preprocess_ref_layout->addWidget(preprocess_ref_img_label);
  preprocess_ref_layout->addWidget(preprocess_ref_img_pixmap);
  main_layout->addWidget(preprocess_ref_widget, 1, 0);

  preprocess_sensed_widget = new QWidget(main_widget);
  QVBoxLayout *preprocess_sensed_layout = new QVBoxLayout();
  preprocess_sensed_widget->setLayout(preprocess_sensed_layout);
  QLabel *preprocess_sensed_img_label = new QLabel(preprocess_sensed_widget);
  preprocess_sensed_img_pixmap = new QLabel(preprocess_sensed_widget);
  preprocess_sensed_img_label->setText("Preprocessed Sensed Image");
  preprocess_sensed_layout->addWidget(preprocess_sensed_img_label);
  preprocess_sensed_layout->addWidget(preprocess_sensed_img_pixmap);
  main_layout->addWidget(preprocess_sensed_widget, 1, 1);

  //---------

  preprocess(false, ref_img, sensed_img, ref_img_preprocessed, sensed_img_preprocessed);
  process(algo->currentText(), model_box->currentText(), matching_method_box->currentText(), ref_img, sensed_img, ref_img_preprocessed,
          sensed_img_preprocessed);

  main_window.show();

  return application.exec();
}
