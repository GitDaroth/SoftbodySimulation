#include "Camera/CameraController.h"


CameraController::CameraController(std::shared_ptr<Camera> camera) :
	m_camera(camera),
	m_isForwardKeyDown(false),
	m_isBackwardKeyDown(false),
	m_isLeftKeyDown(false),
	m_isRightKeyDown(false),
	m_isBoostKeyDown(false),
	m_isRightMouseButtonDown(false),
	m_movementSpeed(1.f),
	m_movementBoost(2.f),
	m_rotationSpeed(1.f),
	m_yawAngle(0.f),
	m_pitchAngle(0.f),
	m_lastMousePosition(QPoint(-1.f, -1.f))
{
}

CameraController::~CameraController()
{
}

void CameraController::update(float deltaTime)
{
	float moveDistance = m_movementSpeed * deltaTime;
	if (m_isBoostKeyDown)
		moveDistance *= m_movementBoost;

	QVector3D moveDirection(0.f, 0.f, 0.f);
	if (m_isForwardKeyDown)
		moveDirection += m_camera->getViewDirection();
	if (m_isBackwardKeyDown)
		moveDirection -= m_camera->getViewDirection();
	if (m_isRightKeyDown)
		moveDirection += m_camera->getRightDirection();
	if (m_isLeftKeyDown)
		moveDirection -= m_camera->getRightDirection();
	moveDirection.normalize();

	m_camera->setPosition(m_camera->getPosition() + moveDirection * moveDistance);
}

void CameraController::onMouseMoveEvent(QMouseEvent * event)
{
	if (m_isRightMouseButtonDown)
	{
		QPoint currentMousePosition = event->globalPos();

		if (m_lastMousePosition.x() < 0.f && m_lastMousePosition.y() < 0.f)
		{
			m_lastMousePosition = currentMousePosition;
			return;
		}

		QPoint deltaMousePosition = currentMousePosition - m_lastMousePosition;
		m_yawAngle += deltaMousePosition.x() * m_rotationSpeed;
		m_pitchAngle += deltaMousePosition.y() * m_rotationSpeed;
		m_pitchAngle = qMin(qMax(m_pitchAngle, -89.f), 89.f);

		m_camera->setRotation(m_yawAngle, m_pitchAngle, 0.f);

		m_lastMousePosition = currentMousePosition;
	}
}

void CameraController::onMousePressEvent(QMouseEvent * event)
{
	if (event->button() == Qt::RightButton)
		m_isRightMouseButtonDown = true;
}

void CameraController::onMouseReleaseEvent(QMouseEvent * event)
{
	if (event->button() == Qt::RightButton)
	{
		m_isRightMouseButtonDown = false;
		m_lastMousePosition = QPoint(-1.f, -1.f);
	}
}

void CameraController::onKeyPressEvent(QKeyEvent * event)
{
	switch (event->key())
	{
	case Qt::Key_W:
		m_isForwardKeyDown = true;
		break;
	case Qt::Key_A:
		m_isLeftKeyDown = true;
		break;
	case Qt::Key_S:
		m_isBackwardKeyDown = true;
		break;
	case Qt::Key_D:
		m_isRightKeyDown = true;
		break;
	case Qt::Key_Shift:
		m_isBoostKeyDown = true;
		break;
	}
}

void CameraController::onKeyReleaseEvent(QKeyEvent * event)
{
	switch (event->key())
	{
	case Qt::Key_W:
		m_isForwardKeyDown = false;
		break;
	case Qt::Key_A:
		m_isLeftKeyDown = false;
		break;
	case Qt::Key_S:
		m_isBackwardKeyDown = false;
		break;
	case Qt::Key_D:
		m_isRightKeyDown = false;
		break;
	case Qt::Key_Shift:
		m_isBoostKeyDown = false;
		break;
	}
}

float CameraController::getMovementSpeed() const
{
	return m_movementSpeed;
}

float CameraController::getRotationSpeed() const
{
	return m_rotationSpeed;
}

float CameraController::getMovementBoost() const
{
	return m_movementBoost;
}

void CameraController::setMovementSpeed(float movementSpeed)
{
	m_movementSpeed = movementSpeed;
}

void CameraController::setRotationSpeed(float rotationSpeed)
{
	m_rotationSpeed = rotationSpeed;
}

void CameraController::setMovementBoost(float movementBoost)
{
	m_movementBoost = movementBoost;
}
