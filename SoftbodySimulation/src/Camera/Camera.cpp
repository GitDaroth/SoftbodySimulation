#include "Camera/Camera.h"

Camera::Camera() :
	m_position(QVector3D(0.f, 0.f, 0.f)),
	m_nearPlane(0.1f),
	m_farPlane(1000.f),
	m_aspectRatio(4.f / 3.f),
	m_fieldOfView(45.f),
	m_isViewMatrixCalcNeeded(true),
	m_isProjectionMatrixCalcNeeded(true)
{
	resetRotation();
}

Camera::~Camera()
{
}

QVector3D Camera::getPosition() const
{
	return m_position;
}

QVector3D Camera::getViewDirection() const
{
	return -m_zAxis;
}

QVector3D Camera::getUpDirection() const
{
	return m_yAxis;
}

QVector3D Camera::getRightDirection() const
{
	return m_xAxis;
}

float Camera::getNearPlane() const
{
	return m_nearPlane;
}

float Camera::getFarPlane() const
{
	return m_farPlane;
}

float Camera::getFieldOfView() const
{
	return m_fieldOfView;
}

float Camera::getAspectRatio() const
{
	return m_aspectRatio;
}

QMatrix4x4 Camera::getViewMatrix()
{
	if (m_isViewMatrixCalcNeeded)
		calcViewMatrix();
	return m_viewMatrix;
}

QMatrix4x4 Camera::getProjectionMatrix()
{
	if (m_isProjectionMatrixCalcNeeded)
		calcProjectionMatrix();
	return m_projectionMatrix;
}

void Camera::setPosition(const QVector3D & position)
{
	m_position = position;
	m_isViewMatrixCalcNeeded = true;
}

void Camera::setRotation(float yaw, float pitch, float roll)
{
	resetRotation();

	QMatrix4x4 rotationMatrix;
	rotationMatrix.setToIdentity();
	rotationMatrix.rotate(-yaw, m_yAxis);
	m_xAxis = rotationMatrix * m_xAxis;
	m_zAxis = rotationMatrix * m_zAxis;

	rotationMatrix.setToIdentity();
	rotationMatrix.rotate(-pitch, m_xAxis);
	m_yAxis = rotationMatrix * m_yAxis;
	m_zAxis = rotationMatrix * m_zAxis;

	rotationMatrix.setToIdentity();
	rotationMatrix.rotate(-roll, m_zAxis);
	m_xAxis = rotationMatrix * m_xAxis;
	m_yAxis = rotationMatrix * m_yAxis;

	orthogonalizeAxes();

	m_isViewMatrixCalcNeeded = true;
}

void Camera::setNearPlane(float nearPlane)
{
	m_nearPlane = nearPlane;
	m_isProjectionMatrixCalcNeeded = true;
}

void Camera::setFarPlane(float farPlane)
{
	m_farPlane = farPlane;
	m_isProjectionMatrixCalcNeeded = true;
}

void Camera::setFieldOfView(float fov)
{
	m_fieldOfView = fov;
	m_isProjectionMatrixCalcNeeded = true;
}

void Camera::setAspectRatio(float aspectRatio)
{
	m_aspectRatio = aspectRatio;
	m_isProjectionMatrixCalcNeeded = true;
}

void Camera::resetRotation()
{
	m_xAxis = QVector3D(1.f, 0.f, 0.f);
	m_yAxis = QVector3D(0.f, 1.f, 0.f);
	m_zAxis = QVector3D(0.f, 0.f, 1.f);
	m_isViewMatrixCalcNeeded = true;
}

void Camera::lookAt(const QVector3D & targetPos, const QVector3D & upDirection)
{
	m_zAxis = -(targetPos - m_position); // View direction = -Z
	m_zAxis.normalize();

	m_xAxis = QVector3D::crossProduct(upDirection, m_zAxis);
	m_xAxis.normalize();

	m_yAxis = QVector3D::crossProduct(m_zAxis, m_xAxis);
	m_yAxis.normalize();

	m_isViewMatrixCalcNeeded = true;
}

void Camera::calcViewMatrix()
{
	m_viewMatrix.setToIdentity();
	m_viewMatrix.lookAt(m_position, m_position + getViewDirection(), getUpDirection());

	m_isViewMatrixCalcNeeded = false;
}

void Camera::calcProjectionMatrix()
{
	m_projectionMatrix.setToIdentity();
	m_projectionMatrix.perspective(m_fieldOfView, m_aspectRatio, m_nearPlane, m_farPlane);

	m_isProjectionMatrixCalcNeeded = false;
}

void Camera::orthogonalizeAxes()
{
	m_zAxis.normalize();

	m_yAxis = QVector3D::crossProduct(m_zAxis, m_xAxis);
	m_yAxis.normalize();

	m_xAxis = QVector3D::crossProduct(m_yAxis, m_zAxis);
	m_xAxis.normalize();
}
