#pragma once

#include "Constraint/Constraint.h"
#include "PBDParticle.h"

#include <QVector>
#include <Eigen/Core>

class ShapeConstraint : public Constraint
{
public:
	ShapeConstraint(const QString& dataPath, const QString& meshPath);
	ShapeConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath);
	~ShapeConstraint();

	void initialize(bool useClusters = true);

	void setStartPosition(const QVector3D& startPosition);
	void setRotation(const Eigen::Matrix3f& rotation);
	virtual void setStiffness(float stiffness) override;
	void setBeta(float beta);

	const QString& getMeshPath();
	const QVector3D& getStartPosition();
	const QVector<std::shared_ptr<PBDParticle>>& getParticles();
	const QVector<std::shared_ptr<ShapeConstraint>>& getClusters();
	bool hasClusters();
	const QVector3D& getInitialCenterOfMass();
	const Eigen::Matrix3f& getRotation();
	const Eigen::Matrix3f& getAqq();
	float getBeta();

	void addParticleIndex(int particleIndex);
	void clearParticleIndices();
	const QVector<int>& getParticleIndices();

protected:
	void readShape(const QString& dataPath);
	void createClusters();
	virtual void calcInitializationData();
	QVector3D calcCenterOfMass(bool initial = false);
	void calcAqq();

	virtual std::shared_ptr<ShapeConstraint> createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles) = 0;

	QString m_meshPath;
	QVector<std::shared_ptr<PBDParticle>> m_particles;
	QVector<std::shared_ptr<ShapeConstraint>> m_clusters;
	QVector3D m_startPosition;
	QVector3D m_initialCenterOfMass;
	float m_beta = 0.8f;
	Eigen::Matrix3f m_rotation;
	Eigen::Matrix3f m_Aqq;

	QVector<int> m_particleIndices;
};