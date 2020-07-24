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

const int size = 400;

Mat ref_img, sensed_img, warped_img, ref_img_preprocessed, sensed_img_preprocessed;
bool preprocessing = false;

QWidget *match_widget;
QWidget *cross_widget;
QWidget *preprocess_ref_widget;
QWidget *preprocess_sensed_widget;
QWidget *comparative_widget;
QWidget *corr_widget;
QWidget *orb_widget;
QWidget *akaze_widget;
QWidget *fmt_widget;

QLabel *warp_img_pixmap;
QLabel *match_img_pixmap;
QLabel *preprocess_ref_img_pixmap;
QLabel *preprocess_sensed_img_pixmap;
QLabel *sensed_img_pixmap;
QLabel *cross_img_pixmap;
QLabel *corr_img_pixmap;
QLabel *orb_img_pixmap;
QLabel *akaze_img_pixmap;
QLabel *fmt_img_pixmap;

QComboBox *model_box;
QComboBox *matching_method_box;
QPushButton *matches_btn;
QPushButton *cross_btn;
QDoubleSpinBox *deriche_gamma;

//------------------------------------------------------------------------------
void preprocess() {

  ref_img_preprocessed = ref_img.clone();
  sensed_img_preprocessed = sensed_img.clone();

  preprocess_ref_widget->setVisible(preprocessing);
  preprocess_sensed_widget->setVisible(preprocessing);
  deriche_gamma->setEnabled(preprocessing);

  if (preprocessing) {
    DericheGradient(ref_img_preprocessed, ref_img_preprocessed, deriche_gamma->value());
    DericheGradient(sensed_img_preprocessed, sensed_img_preprocessed, deriche_gamma->value());
    preprocess_ref_img_pixmap->setPixmap(QPixmap::fromImage(QImage(ref_img_preprocessed.data, ref_img_preprocessed.cols, ref_img_preprocessed.rows,
                                                                   ref_img_preprocessed.step, QImage::Format_RGB888))
                                             .scaled(size, size, Qt::KeepAspectRatio));
    preprocess_sensed_img_pixmap->setPixmap(
        QPixmap::fromImage(QImage(sensed_img_preprocessed.data, sensed_img_preprocessed.cols, sensed_img_preprocessed.rows,
                                  sensed_img_preprocessed.step, QImage::Format_RGB888))
            .scaled(size, size, Qt::KeepAspectRatio));
  }
}

//------------------------------------------------------------------------------
void process(const QString &method, const QString &model, const QString &features_matching) {

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

  double alpha = 0.5;
  double beta = (1.0 - alpha);
  cv::Mat cross_img;
  addWeighted(ref_img, alpha, warp_img, beta, 0.0, cross_img);
  cross_img_pixmap->setPixmap(QPixmap::fromImage(QImage(cross_img.data, cross_img.cols, cross_img.rows, cross_img.step, QImage::Format_RGB888))
                                  .scaled(size * 3, size * 3, Qt::KeepAspectRatio));

  cross_btn->setEnabled(true);

  std::cout << "done" << std::endl;
}

//------------------------------------------------------------------------------
void compareShiftRegistration(bool correlation_is_checked, bool orb_is_checked, bool akaze_is_checked, bool fmt_is_checked, int occurrences_nb) {

  clock_t t0;
  double duration;

  corr_widget->setVisible(false);
  orb_widget->setVisible(false);
  akaze_widget->setVisible(false);
  fmt_widget->setVisible(false);

  if (correlation_is_checked) { // enhancedCorrelationCoefficientMaximization
    Mat corr_translation = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_TRANSLATION);
    cout << "ECC (TRANSLATION MODEL): ---------------------------------" << endl;
    cout << "t[x, y] : " << extractTranslationFromAffine(corr_translation) << endl;
    t0 = clock();
    for (int i = 0; i <= occurrences_nb; i++) {
      Mat corr_translation = enhancedCorrelationCoefficientMaximization(ref_img, sensed_img, MOTION_TRANSLATION);
    }
    duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
    cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

    corr_widget->setVisible(true);
    cv::Mat warp_img = imgRegistration(ref_img, sensed_img, corr_translation);
    corr_img_pixmap->setPixmap(QPixmap::fromImage(QImage(warp_img.data, warp_img.cols, warp_img.rows, warp_img.step, QImage::Format_RGB888))
                                   .scaled(size, size, Qt::KeepAspectRatio));
  }

  if (orb_is_checked) { // Features features methods
    FBConfig config_orb_rigid;
    config_orb_rigid.model = FBConfig::AFFINE_PARTIAL;
    config_orb_rigid.detectorDescriptor = FBConfig::ORB_ALGO;
    Mat tr_orb_rigid = featuresBasedMethod(ref_img, sensed_img, config_orb_rigid);
    cout << "ORB (RIGID MODEL) : --------------------------------------" << endl;
    cout << "t[x, y] : " << extractTranslationFromAffine(tr_orb_rigid) << endl;
    t0 = clock();
    for (int i = 0; i <= occurrences_nb; i++) {
      Mat tr_orb_rigid = featuresBasedMethod(ref_img, sensed_img, config_orb_rigid);
    }
    duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
    cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

    orb_widget->setVisible(true);
    cv::Mat warp_img = imgRegistration(ref_img, sensed_img, tr_orb_rigid);
    orb_img_pixmap->setPixmap(QPixmap::fromImage(QImage(warp_img.data, warp_img.cols, warp_img.rows, warp_img.step, QImage::Format_RGB888))
                                  .scaled(size, size, Qt::KeepAspectRatio));
  }

  if (akaze_is_checked) { // Features features methods
    FBConfig config_akaze_rigid;
    config_akaze_rigid.model = FBConfig::AFFINE_PARTIAL;
    config_akaze_rigid.detectorDescriptor = FBConfig::AKAZE_ALGO;
    Mat tr_akaze_rigid = featuresBasedMethod(ref_img, sensed_img, config_akaze_rigid);
    cout << "AKAZE (RIGID MODEL) : ------------------------------------" << endl;
    cout << "t[x, y] : " << extractTranslationFromAffine(tr_akaze_rigid) << endl;
    t0 = clock();
    for (int i = 0; i <= occurrences_nb; i++) {
      Mat tr_akaze_rigid = featuresBasedMethod(ref_img, sensed_img, config_akaze_rigid);
    }
    duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
    cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

    akaze_widget->setVisible(true);
    cv::Mat warp_img = imgRegistration(ref_img, sensed_img, tr_akaze_rigid);
    akaze_img_pixmap->setPixmap(QPixmap::fromImage(QImage(warp_img.data, warp_img.cols, warp_img.rows, warp_img.step, QImage::Format_RGB888))
                                    .scaled(size, size, Qt::KeepAspectRatio));
  }

  if (fmt_is_checked) { // Fourier-mellin transformation
    Mat fmt = fourierMellinTransform(ref_img, sensed_img);
    cout << "FOURIER-MELLIN (AFFINE MODEL): ---------------------------" << endl;
    cout << "t[x, y] : " << extractTranslationFromAffine(fmt) << endl;
    t0 = clock();
    for (int i = 0; i <= occurrences_nb; i++) {
      Mat fmt = fourierMellinTransform(ref_img, sensed_img);
    }
    duration = ((clock() - t0) / (double)CLOCKS_PER_SEC) / occurrences_nb;
    cout << "mean execution time on " << occurrences_nb << " runs : " << duration << " secondes" << endl;

    fmt_widget->setVisible(true);
    cv::Mat warp_img = imgRegistration(ref_img, sensed_img, fmt);
    fmt_img_pixmap->setPixmap(QPixmap::fromImage(QImage(warp_img.data, warp_img.cols, warp_img.rows, warp_img.step, QImage::Format_RGB888))
                                  .scaled(size, size, Qt::KeepAspectRatio));
  }
}

//------------------------------------------------------------------------------
void shift(const cv::Mat &img, cv::Mat &img_shifted, double x, double y) {
  cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, x, 0, 1, y);
  cv::warpAffine(img, img_shifted, M, img.size(), INTER_LINEAR, BORDER_REPLICATE);
  sensed_img_pixmap->setPixmap(
      QPixmap::fromImage(QImage(img_shifted.data, img_shifted.cols, img_shifted.rows, img_shifted.step, QImage::Format_RGB888))
          .scaled(size, size, Qt::KeepAspectRatio));
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

//----------------------------------------------------------------------------//
int main(int argc, char **argv) {

  // Declare the supported options.
  options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");

  desc.add_options()("registration,R", value<vector<string>>()->multitoken(), "image registration : [reference image path] [sensed image path]");
  desc.add_options()("benchmark,B", value<vector<string>>()->multitoken(),
                     "compare accuracy and execution time of registration methods on ground state computed : [reference image path]");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  // --------------------------------------------------------------------------
  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  // --------------------------------------------------------------------------
  bool benchmark = false;
  vector<string> arg;

  if (vm.count("registration")) {
    arg = vm["registration"].as<vector<string>>();
    if (arg.size() < 2) {
      cout << "Expected at least 2 arguments : [reference image path] [sensed image path]" << endl;
      return -1;
    }
    ref_img = imread(arg[0]);
    if (ref_img.empty()) {
      cout << "Could not open or find the [reference image path]" << endl;
      return -1;
    }
    sensed_img = imread(arg[1]);
    if (sensed_img.empty()) {
      cout << "Could not open or find the [sensed image path]" << endl;
      return -1;
    }
  } else if (vm.count("benchmark")) {
    arg = vm["benchmark"].as<vector<string>>();
    if (arg.size() < 1) {
      cout << "Expected at least 1 argument : [reference image path]" << endl;
      return -1;
    }
    ref_img = imread(arg[0]);
    if (ref_img.empty()) {
      cout << "Could not open or find the [reference image path]" << endl;
      return -1;
    }
    sensed_img = ref_img.clone();
    benchmark = true;
  } else {
    cout << "See --help for more details." << endl;
    return -1;
  }

  ref_img_preprocessed = ref_img.clone();
  sensed_img_preprocessed = sensed_img.clone();

  bool correlation_is_checked = false;
  bool orb_is_checked = true;
  bool akaze_is_checked = false;
  bool fmt_is_checked = true;
  double tr_x = 0;
  double tr_y = 0;

  QString registration_algo = "ORB";
  QString registration_model = "RIGID";
  QString matching_method = "RANSAC";

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

  if (benchmark) {
    QDoubleSpinBox *translation_x = new QDoubleSpinBox(toolbar_registration);
    translation_x->setRange(0, 1000);
    translation_x->setPrefix("x:");
    translation_x->setSingleStep(1);
    translation_x->setValue(tr_x);
    toolbar_registration->addWidget(translation_x);
    QObject::connect(translation_x, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double d) {
      tr_x = d;
      shift(ref_img, sensed_img, tr_x, tr_y);
      preprocess();
    });

    QDoubleSpinBox *translation_y = new QDoubleSpinBox(toolbar_registration);
    translation_y->setRange(0, 1000);
    translation_y->setPrefix("y:");
    translation_y->setSingleStep(1);
    translation_y->setValue(tr_y);
    toolbar_registration->addWidget(translation_y);
    QObject::connect(translation_y, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double d) {
      tr_y = d;
      shift(ref_img, sensed_img, tr_x, tr_y);
      preprocess();
    });

    QCheckBox *correlation_btn = new QCheckBox(toolbar_registration);
    toolbar_registration->addWidget(correlation_btn);
    correlation_btn->setText("Correlation");
    correlation_btn->setChecked(correlation_is_checked);
    QObject::connect(correlation_btn, &QCheckBox::stateChanged, [&](int state) { correlation_is_checked = (state == 0) ? false : true; });

    QCheckBox *orb_btn = new QCheckBox(toolbar_registration);
    toolbar_registration->addWidget(orb_btn);
    orb_btn->setText("ORB");
    orb_btn->setChecked(orb_is_checked);
    QObject::connect(orb_btn, &QCheckBox::stateChanged, [&](int state) { orb_is_checked = (state == 0) ? false : true; });

    QCheckBox *akaze_btn = new QCheckBox(toolbar_registration);
    toolbar_registration->addWidget(akaze_btn);
    akaze_btn->setText("AKAZE");
    akaze_btn->setChecked(akaze_is_checked);
    QObject::connect(akaze_btn, &QCheckBox::stateChanged, [&](int state) { akaze_is_checked = (state == 0) ? false : true; });

    QCheckBox *fmt_btn = new QCheckBox(toolbar_registration);
    toolbar_registration->addWidget(fmt_btn);
    fmt_btn->setText("Fourier-Mellin");
    fmt_btn->setChecked(fmt_is_checked);
    QObject::connect(fmt_btn, &QCheckBox::stateChanged, [&](int state) { fmt_is_checked = (state == 0) ? false : true; });

    QPushButton *process_btn = new QPushButton(toolbar_registration);
    toolbar_registration->addWidget(process_btn);
    process_btn->setText("Compare");
    QObject::connect(process_btn, &QPushButton::clicked,
                     [&]() { compareShiftRegistration(correlation_is_checked, orb_is_checked, akaze_is_checked, fmt_is_checked, 100); });

    toolbar_registration->addSeparator();

    QPushButton *result_btn = new QPushButton(toolbar_registration);
    toolbar_registration->addWidget(result_btn);
    result_btn->setText("Display results");
    QObject::connect(result_btn, &QPushButton::clicked, [&]() { comparative_widget->setVisible(true); });
    result_btn->setEnabled(true);

  } else {
    QComboBox *registration_algo_box = new QComboBox(toolbar_registration);
    toolbar_registration->addWidget(registration_algo_box);
    registration_algo_box->addItem("Correlation");
    registration_algo_box->addItem("ORB");
    registration_algo_box->addItem("AKAZE");
    registration_algo_box->addItem("Fourier-Mellin");
    registration_algo_box->setCurrentIndex(1);
    QObject::connect(registration_algo_box, QOverload<const QString &>::of(&QComboBox::currentTextChanged), [&](const QString &text) {
      registration_algo = text;
      registrationMethodChanged(text);
    });

    model_box = new QComboBox(toolbar_registration);
    toolbar_registration->addWidget(model_box);
    model_box->addItem("HOMOGRAPHY");
    model_box->addItem("AFFINE");
    model_box->addItem("RIGID");
    model_box->addItem("TRANSLATION");
    model_box->setCurrentIndex(2);
    SetComboBoxItemEnabled(model_box, 3, false);
    QObject::connect(model_box, QOverload<const QString &>::of(&QComboBox::currentTextChanged),
                     [&](const QString &text) { registration_model = text; });

    matching_method_box = new QComboBox(toolbar_registration);
    toolbar_registration->addWidget(matching_method_box);
    matching_method_box->addItem("RANSAC");
    matching_method_box->addItem("LMEDS");
    matching_method_box->setCurrentIndex(0);
    QObject::connect(matching_method_box, QOverload<const QString &>::of(&QComboBox::currentTextChanged),
                     [&](const QString &text) { matching_method = text; });

    toolbar_registration->addSeparator();

    QPushButton *process_btn = new QPushButton(toolbar_registration);
    toolbar_registration->addWidget(process_btn);
    process_btn->setText("Process");
    QObject::connect(process_btn, &QPushButton::clicked, [&]() { process(registration_algo, registration_model, matching_method); });

    toolbar_registration->addSeparator();

    matches_btn = new QPushButton(toolbar_registration);
    toolbar_registration->addWidget(matches_btn);
    matches_btn->setText("Display matches");
    QObject::connect(matches_btn, &QPushButton::clicked, [&]() { match_widget->setVisible(true); });
    matches_btn->setEnabled(false);

    cross_btn = new QPushButton(toolbar_registration);
    toolbar_registration->addWidget(cross_btn);
    cross_btn->setText("Display cross-image");
    QObject::connect(cross_btn, &QPushButton::clicked, [&]() { cross_widget->setVisible(true); });
    cross_btn->setEnabled(false);
  }

  //----------TOOLBAR PREPROCESS

  QToolBar *toolbar_preprocess = main_window.addToolBar("toolbar for preprocessing");
  main_window.insertToolBarBreak(toolbar_preprocess);

  QCheckBox *preprocess_btn = new QCheckBox(toolbar_preprocess);
  toolbar_preprocess->addWidget(preprocess_btn);
  preprocess_btn->setText("Preprocess");
  QObject::connect(preprocess_btn, &QCheckBox::stateChanged, [&](int state) {
    preprocessing = (state == 0) ? false : true;
    preprocess();
    main_window.adjustSize();
  });

  deriche_gamma = new QDoubleSpinBox(toolbar_preprocess);
  deriche_gamma->setRange(0, 3);
  deriche_gamma->setPrefix("gamma:");
  deriche_gamma->setSingleStep(0.25);
  deriche_gamma->setValue(0.5);
  deriche_gamma->setEnabled(false);
  toolbar_preprocess->addWidget(deriche_gamma);
  QObject::connect(deriche_gamma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double d) { preprocess(); });

  //----------DISPLAY IMG

  QWidget *reference_widget = new QWidget(main_widget);
  QVBoxLayout *ref_layout = new QVBoxLayout();
  reference_widget->setLayout(ref_layout);
  QLabel *ref_img_label = new QLabel(reference_widget);
  QLabel *ref_img_pixmap = new QLabel(reference_widget);
  ref_img_label->setText("Reference image");
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
  sensed_img_pixmap = new QLabel(sensed_widget);
  sensed_img_label->setText("Sensed image");
  cvtColor(sensed_img, sensed_img, COLOR_BGR2RGB);
  sensed_img_pixmap->setPixmap(QPixmap::fromImage(QImage(sensed_img.data, sensed_img.cols, sensed_img.rows, sensed_img.step, QImage::Format_RGB888))
                                   .scaled(size, size, Qt::KeepAspectRatio));
  sensed_layout->addWidget(sensed_img_label);
  sensed_layout->addWidget(sensed_img_pixmap);
  main_layout->addWidget(sensed_widget, 0, 1);

  if (!benchmark) {
    QWidget *warp_widget = new QWidget(main_widget);
    QVBoxLayout *warp_layout = new QVBoxLayout();
    warp_widget->setLayout(warp_layout);
    QLabel *warp_img_label = new QLabel(warp_widget);
    warp_img_pixmap = new QLabel(warp_widget);
    warp_img_label->setText("Sensed image warped");
    warp_img_label->setAlignment(Qt::AlignCenter);
    warp_layout->addWidget(warp_img_label);
    warp_layout->addWidget(warp_img_pixmap);
    main_layout->addWidget(warp_widget, 0, 2, 2, 2);
  }

  //----------DISPLAY REGISTRATION METHODS COMPARAISON

  if (benchmark) {
    comparative_widget = new QWidget();
    QHBoxLayout *comparative_layout = new QHBoxLayout();
    comparative_widget->setLayout(comparative_layout);

    corr_widget = new QWidget(comparative_widget);
    QVBoxLayout *corr_layout = new QVBoxLayout();
    corr_widget->setLayout(corr_layout);
    QLabel *corr_img_label = new QLabel(corr_widget);
    corr_img_label->setText("Correlation");
    corr_layout->addWidget(corr_img_label);
    corr_img_pixmap = new QLabel(corr_widget);
    corr_layout->addWidget(corr_img_pixmap);
    comparative_layout->addWidget(corr_widget);

    orb_widget = new QWidget(comparative_widget);
    QVBoxLayout *orb_layout = new QVBoxLayout();
    orb_widget->setLayout(orb_layout);
    QLabel *orb_img_label = new QLabel(orb_widget);
    orb_img_label->setText("ORB");
    orb_layout->addWidget(orb_img_label);
    orb_img_pixmap = new QLabel(orb_widget);
    orb_layout->addWidget(orb_img_pixmap);
    comparative_layout->addWidget(orb_widget);

    akaze_widget = new QWidget(comparative_widget);
    QVBoxLayout *akaze_layout = new QVBoxLayout();
    akaze_widget->setLayout(akaze_layout);
    QLabel *akaze_img_label = new QLabel(akaze_widget);
    akaze_img_label->setText("AKAZE");
    akaze_layout->addWidget(akaze_img_label);
    akaze_img_pixmap = new QLabel(akaze_widget);
    akaze_layout->addWidget(akaze_img_pixmap);
    comparative_layout->addWidget(akaze_widget);

    fmt_widget = new QWidget(comparative_widget);
    QVBoxLayout *fmt_layout = new QVBoxLayout();
    fmt_widget->setLayout(fmt_layout);
    QLabel *fmt_img_label = new QLabel(fmt_widget);
    fmt_img_label->setText("Fourier-Mellin");
    fmt_layout->addWidget(fmt_img_label);
    fmt_img_pixmap = new QLabel(fmt_widget);
    fmt_layout->addWidget(fmt_img_pixmap);
    comparative_layout->addWidget(fmt_widget);
  }

  //----------MATCH IMAGE FOR FEATURES BASED REGISTRATION ONLY

  if (!benchmark) {
    match_widget = new QWidget();
    QVBoxLayout *match_layout = new QVBoxLayout();
    match_widget->setLayout(match_layout);
    QLabel *match_img_label = new QLabel(match_widget);
    match_img_pixmap = new QLabel(match_widget);
    match_img_label->setText("Match image (features based methods registration)");
    match_layout->addWidget(match_img_label);
    match_layout->addWidget(match_img_pixmap);
  }

  //----------CROSS DISSOLVE TO SEE RESULT

  if (!benchmark) {
    cross_widget = new QWidget();
    QVBoxLayout *cross_layout = new QVBoxLayout();
    cross_widget->setLayout(cross_layout);
    QLabel *cross_img_label = new QLabel(cross_widget);
    cross_img_pixmap = new QLabel(cross_widget);
    cross_img_label->setText("Cross image");
    cross_layout->addWidget(cross_img_label);
    cross_layout->addWidget(cross_img_pixmap);
  }

  //----------PREPROCESSED IMAGE

  preprocess_ref_widget = new QWidget(main_widget);
  QVBoxLayout *preprocess_ref_layout = new QVBoxLayout();
  preprocess_ref_widget->setLayout(preprocess_ref_layout);
  QLabel *preprocess_ref_img_label = new QLabel(preprocess_ref_widget);
  preprocess_ref_img_pixmap = new QLabel(preprocess_ref_widget);
  preprocess_ref_img_label->setText("Preprocessed reference image");
  preprocess_ref_layout->addWidget(preprocess_ref_img_label);
  preprocess_ref_layout->addWidget(preprocess_ref_img_pixmap);
  main_layout->addWidget(preprocess_ref_widget, 1, 0);

  preprocess_sensed_widget = new QWidget(main_widget);
  QVBoxLayout *preprocess_sensed_layout = new QVBoxLayout();
  preprocess_sensed_widget->setLayout(preprocess_sensed_layout);
  QLabel *preprocess_sensed_img_label = new QLabel(preprocess_sensed_widget);
  preprocess_sensed_img_pixmap = new QLabel(preprocess_sensed_widget);
  preprocess_sensed_img_label->setText("Preprocessed sensed image");
  preprocess_sensed_layout->addWidget(preprocess_sensed_img_label);
  preprocess_sensed_layout->addWidget(preprocess_sensed_img_pixmap);
  main_layout->addWidget(preprocess_sensed_widget, 1, 1);

  preprocess();
  if (!benchmark) {
    process(registration_algo, registration_model, matching_method);
  }
  main_window.show();

  return application.exec();
}
