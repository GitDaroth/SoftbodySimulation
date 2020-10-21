#include "ParticleDragger.h"

#include <QApplication>
#include <QWidget>
#include <QDebug>

ParticleDragger::ParticleDragger(std::shared_ptr<PBDSolver> pbdSolver, std::shared_ptr<Camera> camera) :
	m_pbdSolver(pbdSolver),
	m_camera(camera),
	m_isLeftMouseButtonDown(false)
{
}

ParticleDragger::~ParticleDragger()
{
}

void ParticleDragger::onMouseMoveEvent(QMouseEvent * event)
{
	if (m_isLeftMouseButtonDown)
		m_pbdSolver->moveParticle(m_camera->getPosition(), convertScreenCoordsToDirection3D(event->pos()));
}

void ParticleDragger::onMousePressEvent(QMouseEvent * event)
{
	if (event->button() == Qt::LeftButton)
	{
		m_isLeftMouseButtonDown = true;
		m_pbdSolver->selectParticle(m_camera->getPosition(), convertScreenCoordsToDirection3D(event->pos()));
	}
}

void ParticleDragger::onMouseReleaseEvent(QMouseEvent * event)
{
	if (event->button() == Qt::LeftButton)
	{
		m_isLeftMouseButtonDown = false;
		m_pbdSolver->deselectParticle();
	}
}

QVector3D ParticleDragger::convertScreenCoordsToDirection3D(QPoint screenCoords)
{
	QSize windowSize = QApplication::activeWindow()->size();

	screenCoords.setX(qMax(qMin(screenCoords.x(), windowSize.width()), 0));
	screenCoords.setY(qMax(qMin(screenCoords.y(), windowSize.height()), 0));

	QVector4D ray((2.f * screenCoords.x()) / windowSize.width() - 1.f, 1.f - (2.f * screenCoords.y()) / windowSize.height(), -1.f, 1.f);
	ray = m_camera->getProjectionMatrix().inverted() * ray;
	ray.setZ(-1.f);
	ray.setW(0.f);
	ray = m_camera->getViewMatrix().inverted() * ray;

	QVector3D rayDirection(ray.x(), ray.y(), ray.z());
	rayDirection.normalize();

	return rayDirection;
}
