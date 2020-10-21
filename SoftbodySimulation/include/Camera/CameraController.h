#pragma once#

#include <QMouseEvent>
#include <QKeyEvent>
#include "Camera/Camera.h"

class CameraController
{
public:
	CameraController(std::shared_ptr<Camera> camera);
	~CameraController();

	void update(float deltaTime);

	void onMouseMoveEvent(QMouseEvent* event);
	void onMousePressEvent(QMouseEvent* event);
	void onMouseReleaseEvent(QMouseEvent* event);
	void onKeyPressEvent(QKeyEvent* event);
	void onKeyReleaseEvent(QKeyEvent* event);

	float getMovementSpeed() const;
	float getRotationSpeed() const;
	float getMovementBoost() const;
	void setMovementSpeed(float movementSpeed);
	void setRotationSpeed(float rotationSpeed);
	void setMovementBoost(float movementBoost);

private:
	std::shared_ptr<Camera> m_camera;

	float m_movementSpeed;
	float m_rotationSpeed;
	float m_movementBoost;

	bool m_isForwardKeyDown;
	bool m_isBackwardKeyDown;
	bool m_isLeftKeyDown;
	bool m_isRightKeyDown;
	bool m_isRightMouseButtonDown;
	bool m_isBoostKeyDown;

	QPoint m_lastMousePosition;
	float m_yawAngle;
	float m_pitchAngle;
};

