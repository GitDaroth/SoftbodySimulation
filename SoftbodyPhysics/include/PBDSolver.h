#pragma once

#include <QVector>

#include "PBDParticle.h"
#include "Constraint/ShapeConstraint.h"
#include "Constraint/BoxConstraint.h"
#include "PBDParameterData.h"
#include "CUDA/ParticleData.h"
#include "CUDA/ShapeData.h"
#include "CUDA/ColliderData.h"
#include "CUDA/VertexData.h"

class PBDSolver
{
public:
	PBDSolver();
	~PBDSolver();

	void update(float deltaTime);
	void prepareGPUBuffers();
	void updateStaticCollisionGPUBuffers();
	
	void addParticle(std::shared_ptr<PBDParticle> particle);
	QVector<std::shared_ptr<PBDParticle>> getParticles();
	void clearParticles();
	void addConstraint(std::shared_ptr<Constraint> constraint);
	void clearConstraints();

	void selectParticle(const QVector3D& rayOrigin, const QVector3D& rayDirection);
	void moveParticle(const QVector3D& rayOrigin, const QVector3D& rayDirection);
	void deselectParticle();

	void setSolverIterations(unsigned int solverIterations);
	void setSubsteps(unsigned int substeps);
	void setGravity(const QVector3D& gravity);
	void setDamping(float damping);
	void setParticleSleepingThreshold(float particleSleepingThreshold);
	void setDeformationThreshold(float deformationThreshold);
	void setRotExtractionIterations(unsigned int rotExtractionIterations);
	void setStiffnessSoftLinear(float stiffness);
	void setStiffnessSoftQuadratic(float stiffness);
	void setBetaSoftLinear(float beta);
	void setBetaSoftQuadratic(float beta);
	void enableVolumeConservationSoftLinear(bool useVolumeConservation);
	void setStaticFriction(float staticFriction);
	void setDynamicFriction(float dynamicFriction);

	float* getParticlePositions();
	int getParticleCount();
	int getShapeCount();
	int getClusterCount();
	float* getVertices();
	int getVertexCount();
	unsigned int* getVertexIndices();
	int getVertexIndexCount();
	QVector<std::shared_ptr<BoxConstraint>> getBoxConstraints();

private:
	void integrate(float deltaTime);
	void solveConstraints();
	void correctParticles(float deltaTime);

	void prepareMeshSkinningBuffers();
	void prepareParticleBuffers(QVector<std::shared_ptr<ShapeConstraint>> shapes);
	void prepareShapeBuffers(QVector<std::shared_ptr<ShapeConstraint>> shapes);
	void prepareStaticCollisionBuffers();

	QVector<std::shared_ptr<PBDParticle>> m_particles;
	QVector<std::shared_ptr<Constraint>> m_constraints;

	std::shared_ptr<ShapeData> m_deviceShapeData;
	std::shared_ptr<ParticleData> m_deviceParticleData;
	std::shared_ptr<ColliderData> m_deviceColliderData;
	std::shared_ptr<VertexData> m_deviceVertexData;
	
	unsigned int* m_vertexIndices;
	int m_vertexIndexCount;
	float* m_vertices;
	int m_vertexCount;
	float* m_particlePositions;

	bool m_isInitialized;
	unsigned int m_substeps;
	PBDParameterData m_parameterData;
};