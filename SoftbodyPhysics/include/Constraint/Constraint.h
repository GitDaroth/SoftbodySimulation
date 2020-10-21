#pragma once

class Constraint
{
public:
	Constraint();

	float getStiffness() const;
	virtual void setStiffness(float stiffness);

protected:
	float calcLinearlyDependentStiffness(unsigned int solverIterations);

	float m_stiffness;
};

