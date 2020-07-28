#ifndef QT_ROI_VIEW_HPP
#define QT_ROI_VIEW_HPP
#include <QLabel>
#include <QMouseEvent>
#include <QObject>
#include <QPointF>
#include <QRectF>
#include <QRubberBand>
#include <QWidget>

class ROIView : public QLabel {
  Q_OBJECT

public:
  explicit ROIView(QWidget *parent = 0);

  ~ROIView();

signals:
  void ROIChanged(QRect roi);

protected:
  void mouseMoveEvent(QMouseEvent *event);

  void mousePressEvent(QMouseEvent *event);

  void mouseReleaseEvent(QMouseEvent *event);

private:
  QRubberBand *roi_rps; // representation of the region of interest
  QPoint origin;        // origine of the region of interest (top left)
};

#endif
