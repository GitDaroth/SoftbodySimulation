#pragma once

#include "Constraint/ShapeConstraint.h"

class SoftShapeQuadraticConstraint : public ShapeConstraint
{
public:
	SoftShapeQuadraticConstraint(const QString& dataPath, const QString& meshPath);
	SoftShapeQuadraticConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath);
	~SoftShapeQuadraticConstraint();

	const Eigen::MatrixXf& getAqqTilde();

protected:
	virtual void calcInitializationData() override;
	virtual std::shared_ptr<ShapeConstraint> createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles) override;

	void calcAqqTilde();

	Eigen::MatrixXf m_AqqTilde;
};

