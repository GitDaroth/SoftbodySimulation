#include "Constraint/BoxConstraint.h"

#include <Eigen/Geometry> 
#include <QtMath>

BoxConstraint::BoxConstraint(const QVector3D& position, const QVector3D& halfDimension, bool isBoundary) :
	m_position(position),
	m_halfDimension(halfDimension),
	m_isBoundary(isBoundary)
{
	setRotation(0.f, QVector3D(1.f, 0.f, 0.f));
}

BoxConstraint::~BoxConstraint()
{
}

void BoxConstraint::setPosition(const QVector3D& position)
{
	m_position = position;
}

void BoxConstraint::setHalfDimension(const QVector3D& halfDimension)
{
	m_halfDimension = halfDimension;
}

void BoxConstraint::setRotation(float angleDegrees, const QVector3D& axis)
{
	m_angle = angleDegrees;
	m_rotationAxis = axis;
	m_rotation = Eigen::AngleAxisf((angleDegrees / 180.f) * M_PI, Eigen::Vector3f(axis.x(), axis.y(), axis.z()));
}

void BoxConstraint::setIsBoundary(bool isBoundary)
{
	m_isBoundary = isBoundary;
}

const QVector3D& BoxConstraint::getPosition()
{
	return m_position;
}

const QVector3D& BoxConstraint::getHalfDimension()
{
	return m_halfDimension;
}

const Eigen::Matrix3f& BoxConstraint::getRotation()
{
	return m_rotation;
}

float BoxConstraint::getAngle()
{
	return m_angle;
}

const QVector3D& BoxConstraint::getRotationAxis()
{
	return m_rotationAxis;
}

bool BoxConstraint::getIsBoundary()
{
	return m_isBoundary;
}
