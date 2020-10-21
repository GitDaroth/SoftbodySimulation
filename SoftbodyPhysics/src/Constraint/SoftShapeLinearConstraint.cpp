#include "Constraint/SoftShapeLinearConstraint.h"

SoftShapeLinearConstraint::SoftShapeLinearConstraint(const QString& dataPath, const QString& meshPath) :
	ShapeConstraint(dataPath, meshPath)
{
}

SoftShapeLinearConstraint::SoftShapeLinearConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath) :
	ShapeConstraint(particles, meshPath)
{
}

SoftShapeLinearConstraint::~SoftShapeLinearConstraint()
{
}

std::shared_ptr<ShapeConstraint> SoftShapeLinearConstraint::createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles)
{
	return std::make_shared<SoftShapeLinearConstraint>(particles, m_meshPath);
}