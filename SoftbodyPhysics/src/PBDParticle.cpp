#include "PBDParticle.h"

PBDParticle::PBDParticle() :
	m_isPositionFixed(false),
	m_mass(1.f),
	m_inverseMass(1.f),
	m_position(QVector3D(0.f, 0.f, 0.f)),
	m_velocity(QVector3D(0.f, 0.f, 0.f)),
	m_predictedPosition(QVector3D(0.f, 0.f, 0.f)),
	m_collidionId(0),
	m_shapeClusterCount(0)
{
}


PBDParticle::~PBDParticle()
{
}

QVector3D PBDParticle::getPredictedPosition()
{
	return m_predictedPosition;
}

QVector3D PBDParticle::getPosition()
{
	return m_position;
}

QVector3D PBDParticle::getInitialPosition()
{
	return m_initialPosition;
}

QVector3D PBDParticle::getVelocity()
{
	return m_velocity;
}

float PBDParticle::getMass()
{
	return m_mass;
}

float PBDParticle::getInverseMass()
{
	return m_inverseMass;
}

bool PBDParticle::getIsPositionFixed()
{
	return m_isPositionFixed;
}

int PBDParticle::getCollisionId()
{
	return m_collidionId;
}

int PBDParticle::getShapeClusterCount()
{
	return m_shapeClusterCount;
}

void PBDParticle::setPredictedPosition(const QVector3D& predictedPosition)
{
	m_predictedPosition = predictedPosition;
}

void PBDParticle::setPosition(const QVector3D& position)
{
	m_position = position;
}

void PBDParticle::setInitialPosition(const QVector3D& initialPosition)
{
	m_initialPosition = initialPosition;
}

void PBDParticle::setVelocity(const QVector3D& velocity)
{
	m_velocity = velocity;
}

void PBDParticle::setMass(float mass)
{
	if (mass <= 0.f)
		mass = 0.000001f;

	m_mass = mass;
	m_inverseMass = 1.0f / m_mass;
}

void PBDParticle::setIsPositionFixed(bool isPositionFixed)
{
	m_isPositionFixed = isPositionFixed;
}

void PBDParticle::setCollisionId(int collisionId)
{
	m_collidionId = collisionId;
}

void PBDParticle::setShapeClusterCount(int shapeClusterCount)
{
	m_shapeClusterCount = shapeClusterCount;
}

void PBDParticle::addShapeIndex(int shapeIndex)
{
	m_shapeIndices.push_back(shapeIndex);
}

void PBDParticle::clearShapeIndices()
{
	m_shapeIndices.clear();
}

const QVector<int>& PBDParticle::getShapeIndices()
{
	return m_shapeIndices;
}