#pragma once

#include "Constraint/ShapeConstraint.h"

class SoftShapeLinearConstraint : public ShapeConstraint
{
public:
	SoftShapeLinearConstraint(const QString& dataPath, const QString& meshPath);
	SoftShapeLinearConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath);
	~SoftShapeLinearConstraint();

protected:
	virtual std::shared_ptr<ShapeConstraint> createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles) override;
};

