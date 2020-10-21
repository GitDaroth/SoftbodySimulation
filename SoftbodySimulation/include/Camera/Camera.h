#pragma once

#include <QVector3D>
#include <QMatrix4x4>

class Camera
{
public:
	Camera();
	~Camera();

	QVector3D getPosition() const;
	QVector3D getViewDirection() const;
	QVector3D getUpDirection() const;
	QVector3D getRightDirection() const;
	float getNearPlane() const;
	float getFarPlane() const;
	float getFieldOfView() const;
	float getAspectRatio() const;
	QMatrix4x4 getViewMatrix();
	QMatrix4x4 getProjectionMatrix();

	void setPosition(const QVector3D& position);
	void setRotation(float yaw, float pitch, float roll);
	void setNearPlane(float nearPlane);
	void setFarPlane(float farPlane);
	void setFieldOfView(float fov);
	void setAspectRatio(float aspectRatio);

	void resetRotation();
	void lookAt(const QVector3D& targetPos, const QVector3D& upDirection = QVector3D(0.f, 1.f, 0.f));

private:
	void calcViewMatrix();
	void calcProjectionMatrix();
	void orthogonalizeAxes();

	QMatrix4x4 m_viewMatrix;
	QMatrix4x4 m_projectionMatrix;

	QVector3D m_position;
	QVector3D m_xAxis, m_yAxis, m_zAxis;
	float m_nearPlane, m_farPlane;
	float m_fieldOfView;
	float m_aspectRatio;

	bool m_isViewMatrixCalcNeeded;
	bool m_isProjectionMatrixCalcNeeded;

};

