#include "Constraint/RigidShapeConstraint.h"

RigidShapeConstraint::RigidShapeConstraint(const QString& dataPath, const QString& meshPath) :
	ShapeConstraint(dataPath, meshPath)
{
}

RigidShapeConstraint::RigidShapeConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath) :
	ShapeConstraint(particles, meshPath)
{
}

RigidShapeConstraint::~RigidShapeConstraint()
{
}

std::shared_ptr<ShapeConstraint> RigidShapeConstraint::createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles)
{
	return std::make_shared<RigidShapeConstraint>(particles, m_meshPath);
}

void RigidShapeConstraint::setStiffness(float stiffness)
{
	ShapeConstraint::setStiffness(1.f);
}