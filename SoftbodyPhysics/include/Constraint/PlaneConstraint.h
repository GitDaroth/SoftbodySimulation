#pragma once
#include "Constraint/Constraint.h"
#include "PBDParticle.h"

class PlaneConstraint : public Constraint
{
public:
	PlaneConstraint(const QVector3D& normal, float distance);
	~PlaneConstraint();

	const QVector3D& getNormal();
	float getDistance();

private:
	QVector3D m_normal;
	float m_distance;
};

