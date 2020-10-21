#include "Constraint/SoftShapeQuadraticConstraint.h"

#include <unsupported/Eigen/MatrixFunctions>

SoftShapeQuadraticConstraint::SoftShapeQuadraticConstraint(const QString& dataPath, const QString& meshPath) :
	ShapeConstraint(dataPath, meshPath)
{
}

SoftShapeQuadraticConstraint::SoftShapeQuadraticConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath) :
	ShapeConstraint(particles, meshPath)
{
}

SoftShapeQuadraticConstraint::~SoftShapeQuadraticConstraint()
{
}

void SoftShapeQuadraticConstraint::calcInitializationData()
{
	ShapeConstraint::calcInitializationData();
	calcAqqTilde();
}

std::shared_ptr<ShapeConstraint> SoftShapeQuadraticConstraint::createCluster(const QVector<std::shared_ptr<PBDParticle>>& particles)
{
	return std::make_shared<SoftShapeQuadraticConstraint>(particles, m_meshPath);
}

void SoftShapeQuadraticConstraint::calcAqqTilde()
{
	Eigen::MatrixXf AqqTildeTemp(9, 9);
	AqqTildeTemp.fill(0.f);
	for (auto particle : m_particles)
	{
		QVector3D q = particle->getInitialPosition() - m_initialCenterOfMass;

		Eigen::MatrixXf mat(9, 9);
		// Erste Zeile
		mat << q.x() * q.x(),
			q.x() * q.y(),
			q.x() * q.z(),
			q.x() * q.x() * q.x(),
			q.x() * q.y() * q.y(),
			q.x() * q.z() * q.z(),
			q.x() * q.x() * q.y(),
			q.x() * q.y() * q.z(),
			q.x() * q.z() * q.x(),
			// Zweite Zeile
			q.y() * q.x(),
			q.y() * q.y(),
			q.y() * q.z(),
			q.y() * q.x() * q.x(),
			q.y() * q.y() * q.y(),
			q.y() * q.z() * q.z(),
			q.y() * q.x() * q.y(),
			q.y() * q.y() * q.z(),
			q.y() * q.z() * q.x(),
			// Dritte Zeile
			q.z() * q.x(),
			q.z() * q.y(),
			q.z() * q.z(),
			q.z() * q.x() * q.x(),
			q.z() * q.y() * q.y(),
			q.z() * q.z() * q.z(),
			q.z() * q.x() * q.y(),
			q.z() * q.y() * q.z(),
			q.z() * q.z() * q.x(),
			// Vierte Zeile
			q.x() * q.x() * q.x(),
			q.x() * q.x() * q.y(),
			q.x() * q.x() * q.z(),
			q.x() * q.x() * q.x() * q.x(),
			q.x() * q.x() * q.y() * q.y(),
			q.x() * q.x() * q.z() * q.z(),
			q.x() * q.x() * q.x() * q.y(),
			q.x() * q.x() * q.y() * q.z(),
			q.x() * q.x() * q.z() * q.x(),
			// Fünfte Zeile
			q.y() * q.y() * q.x(),
			q.y() * q.y() * q.y(),
			q.y() * q.y() * q.z(),
			q.y() * q.y() * q.x() * q.x(),
			q.y() * q.y() * q.y() * q.y(),
			q.y() * q.y() * q.z() * q.z(),
			q.y() * q.y() * q.x() * q.y(),
			q.y() * q.y() * q.y() * q.z(),
			q.y() * q.y() * q.z() * q.x(),
			// Sechste Zeile
			q.z() * q.z() * q.x(),
			q.z() * q.z() * q.y(),
			q.z() * q.z() * q.z(),
			q.z() * q.z() * q.x() * q.x(),
			q.z() * q.z() * q.y() * q.y(),
			q.z() * q.z() * q.z() * q.z(),
			q.z() * q.z() * q.x() * q.y(),
			q.z() * q.z() * q.y() * q.z(),
			q.z() * q.z() * q.z() * q.x(),
			// Siebte Zeile
			q.x() * q.y() * q.x(),
			q.x() * q.y() * q.y(),
			q.x() * q.y() * q.z(),
			q.x() * q.y() * q.x() * q.x(),
			q.x() * q.y() * q.y() * q.y(),
			q.x() * q.y() * q.z() * q.z(),
			q.x() * q.y() * q.x() * q.y(),
			q.x() * q.y() * q.y() * q.z(),
			q.x() * q.y() * q.z() * q.x(),
			// Achte Zeile
			q.y() * q.z() * q.x(),
			q.y() * q.z() * q.y(),
			q.y() * q.z() * q.z(),
			q.y() * q.z() * q.x() * q.x(),
			q.y() * q.z() * q.y() * q.y(),
			q.y() * q.z() * q.z() * q.z(),
			q.y() * q.z() * q.x() * q.y(),
			q.y() * q.z() * q.y() * q.z(),
			q.y() * q.z() * q.z() * q.x(),
			// Neunte Zeile
			q.z() * q.x() * q.x(),
			q.z() * q.x() * q.y(),
			q.z() * q.x() * q.z(),
			q.z() * q.x() * q.x() * q.x(),
			q.z() * q.x() * q.y() * q.y(),
			q.z() * q.x() * q.z() * q.z(),
			q.z() * q.x() * q.x() * q.y(),
			q.z() * q.x() * q.y() * q.z(),
			q.z() * q.x() * q.z() * q.x();

		AqqTildeTemp += particle->getMass() * mat;
	}
	m_AqqTilde = AqqTildeTemp.inverse();
}

const Eigen::MatrixXf& SoftShapeQuadraticConstraint::getAqqTilde()
{
	return m_AqqTilde;
}