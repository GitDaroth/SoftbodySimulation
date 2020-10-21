#include "Constraint/ShapeConstraint.h"

#include <QFile>
#include <QDataStream>
#include <QRandomGenerator>
#include <QMath.h>
#include <unsupported/Eigen/MatrixFunctions>

ShapeConstraint::ShapeConstraint(const QString & dataPath, const QString& meshPath) :
	m_meshPath(meshPath)
{
	readShape(dataPath);
}

ShapeConstraint::ShapeConstraint(const QVector<std::shared_ptr<PBDParticle>>& particles, const QString& meshPath) :
	m_meshPath(meshPath)
{
	m_particles = particles;
}

void ShapeConstraint::initialize(bool useClusters)
{
	m_startPosition = QVector3D(0.f, 0.f, 0.f);

	if (useClusters)
		createClusters();
	else
		calcInitializationData();
}

ShapeConstraint::~ShapeConstraint()
{
}

void ShapeConstraint::readShape(const QString & dataPath)
{
	QFile file(dataPath);
	file.open(QIODevice::ReadOnly);
	QDataStream in(&file);

	int vertexCount;
	in >> vertexCount;

	for (int i = 0; i < vertexCount; i++)
	{
		float x, y, z;
		in >> x >> y >> z;

		std::shared_ptr<PBDParticle> particle = std::make_shared<PBDParticle>();
		particle->setMass(1.f);
		particle->setInitialPosition(QVector3D(x, y, z));
		particle->setPosition(particle->getInitialPosition());
		particle->setVelocity(QVector3D(0.f, 0.f, 0.f));
		m_particles.push_back(particle);
	}

	file.close();
}

void ShapeConstraint::createClusters()
{
	QVector<std::shared_ptr<PBDParticle>> unassignedParticles = m_particles;
	while (!unassignedParticles.empty())
	{
		// pick a random particle in the pool of particles that havn't been assigned yet
		int randomParticleIndex = QRandomGenerator::global()->bounded(0, unassignedParticles.size());
		std::shared_ptr<PBDParticle> randomParticle = unassignedParticles[randomParticleIndex];

		// find all particles that are in the clusterRadius around the random particle
		// and use them to create a new cluster
		float clusterRadius = 0.7f;
		QVector<std::shared_ptr<PBDParticle>> clusterParticles;
		for (auto particle : m_particles)
		{
			if ((particle->getPosition() - randomParticle->getPosition()).length() <= clusterRadius)
			{
				clusterParticles.push_back(particle);
				unassignedParticles.removeOne(particle);
				particle->setShapeClusterCount(particle->getShapeClusterCount() + 1);
			}
		}
		std::shared_ptr<ShapeConstraint> cluster = createCluster(clusterParticles);
		cluster->initialize(false);
		m_clusters.push_back(cluster);
	}
}

void ShapeConstraint::calcInitializationData()
{
	m_rotation.setIdentity();
	m_initialCenterOfMass = calcCenterOfMass(true);
	calcAqq();
}

QVector3D ShapeConstraint::calcCenterOfMass(bool initial)
{
	float totalMass = 0.f;
	QVector3D weightedPosition(0.f, 0.f, 0.f);
	for (auto particle : m_particles)
	{
		totalMass += particle->getMass();
		if(initial)
			weightedPosition += particle->getMass() * particle->getInitialPosition();
		else
			weightedPosition += particle->getMass() * particle->getPredictedPosition();
	}
	return weightedPosition / totalMass;
}

void ShapeConstraint::calcAqq()
{
	Eigen::Matrix3f tempAqq;
	tempAqq.fill(0.f);
	float matValues[9];
	for (auto particle : m_particles)
	{
		QVector3D q = particle->getInitialPosition() - m_initialCenterOfMass;
		matValues[0] = q.x() * q.x();
		matValues[1] = q.y() * q.x();
		matValues[2] = q.z() * q.x();
		matValues[3] = q.x() * q.y();
		matValues[4] = q.y() * q.y();
		matValues[5] = q.z() * q.y();
		matValues[6] = q.x() * q.z();
		matValues[7] = q.y() * q.z();
		matValues[8] = q.z() * q.z();
		tempAqq += particle->getMass() * Eigen::Matrix3f(matValues); // column major per default
	}
	m_Aqq = tempAqq.inverse();
}

void ShapeConstraint::setStartPosition(const QVector3D & startPosition)
{
	m_startPosition = startPosition;

	for (auto particle : m_particles)
		particle->setPosition(particle->getInitialPosition() + m_startPosition);
}

void ShapeConstraint::setRotation(const Eigen::Matrix3f & rotation)
{
	m_rotation = rotation;
}

void ShapeConstraint::setStiffness(float stiffness)
{
	Constraint::setStiffness(stiffness);

	for (auto cluster : m_clusters)
		cluster->setStiffness(stiffness);
}

void ShapeConstraint::setBeta(float beta)
{
	m_beta = beta;

	for (auto cluster : m_clusters)
		cluster->setBeta(beta);
}

const QString& ShapeConstraint::getMeshPath()
{
	return m_meshPath;
}

const QVector3D& ShapeConstraint::getStartPosition()
{
	return m_startPosition;
}

const QVector<std::shared_ptr<PBDParticle>>& ShapeConstraint::getParticles()
{
	return m_particles;
}

const QVector<std::shared_ptr<ShapeConstraint>>& ShapeConstraint::getClusters()
{
	return m_clusters;
}

bool ShapeConstraint::hasClusters()
{
	return !m_clusters.empty();
}

const QVector3D& ShapeConstraint::getInitialCenterOfMass()
{
	return m_initialCenterOfMass;
}

const Eigen::Matrix3f& ShapeConstraint::getRotation()
{
	return m_rotation;
}

const Eigen::Matrix3f& ShapeConstraint::getAqq()
{
	return m_Aqq;
}

float ShapeConstraint::getBeta()
{
	return m_beta;
}

void ShapeConstraint::addParticleIndex(int particleIndex)
{
	m_particleIndices.push_back(particleIndex);
}

void ShapeConstraint::clearParticleIndices()
{
	m_particleIndices.clear();
}

const QVector<int>& ShapeConstraint::getParticleIndices()
{
	return m_particleIndices;
}