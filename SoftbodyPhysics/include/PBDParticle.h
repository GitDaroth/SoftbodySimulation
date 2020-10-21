#pragma once

#include <QVector3D>
#include <QVector>

class PBDParticle
{
public:
	PBDParticle();
	~PBDParticle();

	QVector3D getPredictedPosition();
	QVector3D getPosition();
	QVector3D getInitialPosition();
	QVector3D getVelocity();
	float getMass();
	float getInverseMass();
	bool getIsPositionFixed();
	int getCollisionId();
	int getShapeClusterCount();

	void setPredictedPosition(const QVector3D& predictedPosition);
	void setPosition(const QVector3D& position);
	void setInitialPosition(const QVector3D& initialPosition);
	void setVelocity(const QVector3D& velocity);
	void setMass(float mass);
	void setIsPositionFixed(bool isPositionFixed);
	void setCollisionId(int collisionId);
	void setShapeClusterCount(int shapeClusterCount);

	void addShapeIndex(int shapeIndex);
	void clearShapeIndices();
	const QVector<int>& getShapeIndices();

private:
	QVector3D m_predictedPosition;
	QVector3D m_position;
	QVector3D m_initialPosition;
	QVector3D m_velocity;
	float m_mass;
	float m_inverseMass;
	bool m_isPositionFixed;
	int m_collidionId;
	int m_shapeClusterCount;

	QVector<int> m_shapeIndices;
};

