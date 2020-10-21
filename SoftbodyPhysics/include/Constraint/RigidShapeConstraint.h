#pragma once

#include "Constraint/ShapeConstraint.h"

class RigidShapeConstraint : public ShapeConstraint
{
public:
	RigidShapeConstraint(const QString& dataPath, const QString& meshPath);
	RigidShapeConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath);
	~RigidShapeConstraint();

	virtual void setStiffness(float stiffness) override;

protected:
	virtual std::shared_ptr<ShapeConstraint> createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles) override;
};

