#pragma once

#include <QMouseEvent>
#include "Camera/Camera.h"
#include <PBDSolver.h>

class ParticleDragger
{
public:
	ParticleDragger(std::shared_ptr<PBDSolver> pbdSolver, std::shared_ptr<Camera> camera);
	~ParticleDragger();

	void onMouseMoveEvent(QMouseEvent* event);
	void onMousePressEvent(QMouseEvent* event);
	void onMouseReleaseEvent(QMouseEvent* event);

private:
	QVector3D convertScreenCoordsToDirection3D(QPoint screenCoords);

	std::shared_ptr<PBDSolver> m_pbdSolver;
	std::shared_ptr<Camera> m_camera;
	bool m_isLeftMouseButtonDown;
};

