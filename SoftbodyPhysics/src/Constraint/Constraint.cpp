#include "Constraint/Constraint.h"
#include <QMath.h>

Constraint::Constraint() :
	m_stiffness(1.f)
{
}

float Constraint::getStiffness() const
{
	return m_stiffness;
}

void Constraint::setStiffness(float stiffness)
{
	m_stiffness = qMin(qMax(stiffness, 0.f), 1.f); // stiffness[0..1]
}

float Constraint::calcLinearlyDependentStiffness(unsigned int solverIterations)
{
	return 1.0f - qPow(1.f - m_stiffness, 1.f / (float)solverIterations);
}
