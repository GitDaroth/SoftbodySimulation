#include "Constraint/PlaneConstraint.h"

PlaneConstraint::PlaneConstraint(const QVector3D& normal, float distance) :
	m_normal(normal),
	m_distance(distance)
{
}

PlaneConstraint::~PlaneConstraint()
{
}

const QVector3D& PlaneConstraint::getNormal()
{
	return m_normal;
}

float PlaneConstraint::getDistance()
{
	return m_distance;
}
