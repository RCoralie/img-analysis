#include "ROIView.hpp"
#include <QMouseEvent>
#include <QRect>

ROIView::ROIView(QWidget *parent) : QLabel(parent) {
  roi_rps = new QRubberBand(QRubberBand::Rectangle, this);
  origin = QPoint();
}

ROIView::~ROIView() { delete roi_rps; }

void ROIView::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    origin = event->pos();
    roi_rps->setGeometry(QRect(origin, QSize()));
    roi_rps->show();
  }
}

void ROIView::mouseMoveEvent(QMouseEvent *event) {
  if (!origin.isNull()) {
    roi_rps->setGeometry(QRect(origin, event->pos()));
  }
}

void ROIView::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit ROIChanged(QRect(origin, event->pos()));
  }
  // roi_rps->hide();
}
