#pragma once

#include "Constraint/Constraint.h"
#include <QVector3D>
#include <Eigen/Core>

class BoxConstraint : public Constraint
{
public:
	BoxConstraint(const QVector3D& position, const QVector3D& halfDimension, bool isBoundary = false);
	~BoxConstraint();

	void setPosition(const QVector3D& position);
	void setHalfDimension(const QVector3D& halfDimension);
	void setRotation(float angleDegrees, const QVector3D& axis);
	void setIsBoundary(bool isBoundary);

	const QVector3D& getPosition();
	const QVector3D& getHalfDimension();
	const Eigen::Matrix3f& getRotation();
	float getAngle();
	const QVector3D& getRotationAxis();
	bool getIsBoundary();

private:
	QVector3D m_position;
	QVector3D m_halfDimension;
	Eigen::Matrix3f m_rotation;
	float m_angle;
	QVector3D m_rotationAxis;
	bool m_isBoundary;
};