#include "PBDSolver.h"

#include "CUDA/Integrate.cuh"
#include "CUDA/ConstructGrid.cuh"
#include "CUDA/UpdateShapes.cuh"
#include "CUDA/SolveConstraints.cuh"
#include "CUDA/CorrectParticles.cuh"
#include "CUDA/RaytestParticles.cuh"
#include "CUDA/SkinMesh.cuh"

#include "Constraint/PlaneConstraint.h"
#include "Constraint/BoxConstraint.h"
#include "Constraint/ShapeConstraint.h"
#include "Constraint/RigidShapeConstraint.h"
#include "Constraint/SoftShapeLinearConstraint.h"
#include "Constraint/SoftShapeQuadraticConstraint.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

PBDSolver::PBDSolver()
{
	m_particlePositions = nullptr;
	m_vertices = nullptr;
	m_vertexIndices = nullptr;
	m_vertexCount = 0;
	m_vertexIndexCount = 0;

	setDamping(0.99f);
	setSolverIterations(8);
	setSubsteps(1);
	setParticleSleepingThreshold(0.001f);
	setGravity(QVector3D(0.f, -9.81f, 0.f));
	setDeformationThreshold(5.f);
	setRotExtractionIterations(10);
	setStiffnessSoftLinear(0.7f);
	setStiffnessSoftQuadratic(0.7f);
	setBetaSoftLinear(0.8f);
	setBetaSoftQuadratic(0.8f);
	enableVolumeConservationSoftLinear(true);
	setStaticFriction(0.5f);
	setDynamicFriction(0.3f);

	m_isInitialized = false;
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus == cudaSuccess)
		m_isInitialized = true;
}

PBDSolver::~PBDSolver()
{
}

void PBDSolver::update(float deltaTime)
{
	if (!m_isInitialized)
		return;

	deltaTime /= (float)m_substeps;
	for (int i = 0; i < m_substeps; i++)
	{
		integrate(deltaTime);
		solveConstraints();
		correctParticles(deltaTime);
	}

	skinMeshWithCuda(m_deviceVertexData, m_deviceParticleData, m_deviceShapeData);

	cudaDeviceSynchronize();

	cudaMemcpy(m_vertices, m_deviceVertexData->vertex, 2 * m_deviceVertexData->verticesCount * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_particlePositions, m_deviceParticleData->position, m_particles.size() * sizeof(float3), cudaMemcpyDeviceToHost);
}

void PBDSolver::integrate(float deltaTime)
{
	integrateWithCuda(m_deviceParticleData, m_parameterData, deltaTime);
}

void PBDSolver::solveConstraints()
{
	constructGridWithCuda(m_deviceParticleData, m_parameterData);
	for (int i = 0; i < m_parameterData.solverIterations; i++)
	{
		updateShapesWithCuda(m_deviceShapeData, m_deviceParticleData, m_parameterData);
		solveConstraintsWithCuda(m_deviceParticleData, m_deviceShapeData, m_deviceColliderData, m_parameterData);
		solveParticleCollisionsWithCuda(m_deviceParticleData, m_parameterData);
	}
}

void PBDSolver::correctParticles(float deltaTime)
{
	correctParticlesWithCuda(m_deviceParticleData, m_parameterData, deltaTime);
}

void PBDSolver::addParticle(std::shared_ptr<PBDParticle> particle)
{
	m_particles.push_back(particle);
}

QVector<std::shared_ptr<PBDParticle>> PBDSolver::getParticles()
{
	return m_particles;
}

void PBDSolver::clearParticles()
{
	m_particles.clear();
}

void PBDSolver::addConstraint(std::shared_ptr<Constraint> constraint)
{
	m_constraints.push_back(constraint);
}

void PBDSolver::clearConstraints()
{
	m_constraints.clear();
}

void PBDSolver::selectParticle(const QVector3D& rayOrigin, const QVector3D& rayDirection)
{
	if (!m_isInitialized)
		return;

	raytestParticlesWithCuda(m_deviceParticleData, convertToFloat3(rayOrigin), convertToFloat3(rayDirection));
	cudaDeviceSynchronize();

	float* raytestResult = new float[m_particles.size()];
	cudaMemcpy(raytestResult, m_deviceParticleData->raytestResult, m_particles.size() * sizeof(float), cudaMemcpyDeviceToHost);

	float minRaytestResult = 99999999.f;
	int minParticleIndex = -1;
	for (int i = 0; i < m_particles.size(); i++)
	{
		if (raytestResult[i] > 0.f && raytestResult[i] < minRaytestResult)
		{
			minRaytestResult = raytestResult[i];
			minParticleIndex = i;
		}
	}

	m_deviceParticleData->selectedParticleIndex = minParticleIndex;
	m_deviceParticleData->selectedParticleDistance = minRaytestResult;
	m_deviceParticleData->selectedParticlePosition = convertToFloat3(rayOrigin + rayDirection * m_deviceParticleData->selectedParticleDistance);
}

void PBDSolver::moveParticle(const QVector3D& rayOrigin, const QVector3D& rayDirection)
{
	m_deviceParticleData->selectedParticlePosition = convertToFloat3(rayOrigin + rayDirection * m_deviceParticleData->selectedParticleDistance);
}

void PBDSolver::deselectParticle()
{
	m_deviceParticleData->selectedParticleIndex = -1;
}

void PBDSolver::setSolverIterations(unsigned int solverIterations)
{
	m_parameterData.solverIterations = solverIterations;
}

void PBDSolver::setSubsteps(unsigned int substeps)
{
	m_substeps = substeps;
}

void PBDSolver::setGravity(const QVector3D& gravity)
{
	m_parameterData.gravity = convertToFloat3(gravity);
}

void PBDSolver::setDamping(float damping)
{
	m_parameterData.damping = damping;
}

void PBDSolver::setParticleSleepingThreshold(float particleSleepingThreshold)
{
	m_parameterData.particleSleepingThreshold = particleSleepingThreshold;
}

void PBDSolver::setDeformationThreshold(float deformationThreshold)
{
	m_parameterData.deformationThreshold = deformationThreshold;
}

void PBDSolver::setRotExtractionIterations(unsigned int rotExtractionIterations)
{
	m_parameterData.rotExtractionIterations = rotExtractionIterations;
}

void PBDSolver::setStiffnessSoftLinear(float stiffness)
{
	m_parameterData.stiffness_softLinear = stiffness;
}

void PBDSolver::setStiffnessSoftQuadratic(float stiffness)
{
	m_parameterData.stiffness_softQuadratic = stiffness;
}

void PBDSolver::setBetaSoftLinear(float beta)
{
	m_parameterData.beta_softLinear = beta;
}

void PBDSolver::setBetaSoftQuadratic(float beta)
{
	m_parameterData.beta_softQuadratic = beta;
}

void PBDSolver::enableVolumeConservationSoftLinear(bool useVolumeConservation)
{
	m_parameterData.useVolumeConservation_softLinear = useVolumeConservation;
}

void PBDSolver::setStaticFriction(float staticFriction)
{
	m_parameterData.staticFriction = staticFriction;
}

void PBDSolver::setDynamicFriction(float dynamicFriction)
{
	m_parameterData.dynamicFriction = dynamicFriction;
}

float* PBDSolver::getParticlePositions()
{
	return m_particlePositions;
}

int PBDSolver::getParticleCount()
{
	return m_particles.size();
}

int PBDSolver::getShapeCount()
{
	int shapeCount = 0;
	for (auto constraint : m_constraints)
		if (std::shared_ptr<ShapeConstraint> shapeConstraint = std::dynamic_pointer_cast<ShapeConstraint>(constraint))
			shapeCount++;

	return shapeCount;
}

int PBDSolver::getClusterCount()
{
	return m_deviceShapeData->shapeCount;
}

float* PBDSolver::getVertices()
{
	return m_vertices;
}

int PBDSolver::getVertexCount()
{
	return m_deviceVertexData->verticesCount;
}

unsigned int* PBDSolver::getVertexIndices()
{
	return m_vertexIndices;
}

int PBDSolver::getVertexIndexCount()
{
	return m_vertexIndexCount;
}

QVector<std::shared_ptr<BoxConstraint>> PBDSolver::getBoxConstraints()
{
	QVector<std::shared_ptr<BoxConstraint>> boxes;
	for (auto constraint : m_constraints)
		if (std::shared_ptr<BoxConstraint> boxConstraint = std::dynamic_pointer_cast<BoxConstraint>(constraint))
			boxes.push_back(boxConstraint);

	return boxes;
}

void PBDSolver::prepareGPUBuffers()
{
	if (!m_isInitialized)
		return;

	prepareMeshSkinningBuffers();

	// shapes list including clusters
	int collisionId = 0;
	QVector<std::shared_ptr<ShapeConstraint>> shapes;
	for (auto constraint : m_constraints)
	{
		if (std::shared_ptr<ShapeConstraint> shapeConstraint = std::dynamic_pointer_cast<ShapeConstraint>(constraint))
		{
			if (shapeConstraint->hasClusters())
				for (auto cluster : shapeConstraint->getClusters())
					shapes.push_back(cluster);
			else
				shapes.push_back(shapeConstraint);

			// collision id in order to exclude particles of same shape to collide with each other
			for (auto particle : shapeConstraint->getParticles())
				particle->setCollisionId(collisionId);
			collisionId++;
		}
	}
	prepareParticleBuffers(shapes);
	prepareShapeBuffers(shapes);

	prepareStaticCollisionBuffers();
}

void PBDSolver::prepareMeshSkinningBuffers()
{
	for (auto particle : m_particles)
		particle->clearShapeIndices();

	for (int i = 0; i < m_constraints.size(); i++)
	{
		if (std::shared_ptr<ShapeConstraint> shapeConstraint = std::dynamic_pointer_cast<ShapeConstraint>(m_constraints[i]))
		{
			shapeConstraint->clearParticleIndices();
			for (auto particle : shapeConstraint->getParticles())
				particle->addShapeIndex(i);
		}
	}

	for (int i = 0; i < m_particles.size(); i++)
	{
		for (int shapeIndex : m_particles[i]->getShapeIndices())
		{
			if (std::shared_ptr<ShapeConstraint> shapeConstraint = std::dynamic_pointer_cast<ShapeConstraint>(m_constraints[shapeIndex]))
			{
				shapeConstraint->addParticleIndex(i);
			}
		}
	}

	QVector<unsigned int> vertexIndices;			// size: 3 * #faces
	QVector<QVector3D> vertexPositions;
	QVector<QVector3D> transformedVertexPositions;
	QVector<QVector3D> vertexNormals;
	QVector<unsigned int> particleIndicesPerVertex; // size: 4 * #vertices
	QVector<float> weightsPerVertex;				// size: 4 * #vertices
	QVector<QVector3D> restPositionsPerVertex;		// size: 4 * #vertices

	for (auto constraint : m_constraints)
	{
		if (std::shared_ptr<ShapeConstraint> shapeConstraint = std::dynamic_pointer_cast<ShapeConstraint>(constraint))
		{
			std::string inputfile = shapeConstraint->getMeshPath().toStdString();
			tinyobj::attrib_t attrib;
			std::vector<tinyobj::shape_t> shapes;
			std::vector<tinyobj::material_t> materials;
			std::string warn;
			std::string err;

			bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

			int verticesSize = vertexPositions.size();
			for (int i = 0; i < shapes[0].mesh.indices.size(); i++)
				vertexIndices.push_back(shapes[0].mesh.indices[i].vertex_index + verticesSize);

			for (int i = 0; i < attrib.vertices.size(); i += 3)
			{
				QVector3D vertexPosition(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
				vertexPositions.push_back(vertexPosition);
				transformedVertexPositions.push_back(vertexPosition + shapeConstraint->getStartPosition());
				vertexNormals.push_back(QVector3D(attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2]));

				float minDistances[4] = { 999999.f, 999999.f, 999999.f, 999999.f };
				unsigned int minParticleIndices[4];
				QVector3D minRestPositionsPerVertex[4];
				for (int j = 0; j < shapeConstraint->getParticleIndices().size(); j++)
				{
					unsigned int particleIndex = shapeConstraint->getParticleIndices()[j];
					QVector3D restPositionPerVertex = vertexPosition - m_particles[particleIndex]->getInitialPosition();
					float distance = restPositionPerVertex.length();

					if (distance < minDistances[0])
					{
						minDistances[3] = minDistances[2];
						minParticleIndices[3] = minParticleIndices[2];
						minRestPositionsPerVertex[3] = minRestPositionsPerVertex[2];

						minDistances[2] = minDistances[1];
						minParticleIndices[2] = minParticleIndices[1];
						minRestPositionsPerVertex[2] = minRestPositionsPerVertex[1];

						minDistances[1] = minDistances[0];
						minParticleIndices[1] = minParticleIndices[0];
						minRestPositionsPerVertex[1] = minRestPositionsPerVertex[0];

						minDistances[0] = distance;
						minParticleIndices[0] = particleIndex;
						minRestPositionsPerVertex[0] = restPositionPerVertex;
					}
					else if (distance < minDistances[1])
					{
						minDistances[3] = minDistances[2];
						minParticleIndices[3] = minParticleIndices[2];
						minRestPositionsPerVertex[3] = minRestPositionsPerVertex[2];

						minDistances[2] = minDistances[1];
						minParticleIndices[2] = minParticleIndices[1];
						minRestPositionsPerVertex[2] = minRestPositionsPerVertex[1];

						minDistances[1] = distance;
						minParticleIndices[1] = particleIndex;
						minRestPositionsPerVertex[1] = restPositionPerVertex;
					}
					else if (distance < minDistances[2])
					{
						minDistances[3] = minDistances[2];
						minParticleIndices[3] = minParticleIndices[2];
						minRestPositionsPerVertex[3] = minRestPositionsPerVertex[2];

						minDistances[2] = distance;
						minParticleIndices[2] = particleIndex;
						minRestPositionsPerVertex[2] = restPositionPerVertex;
					}
					else if (distance < minDistances[3])
					{
						minDistances[3] = distance;
						minParticleIndices[3] = particleIndex;
						minRestPositionsPerVertex[3] = restPositionPerVertex;
					}
				}

				float minWeightsPerVertex[4];
				float sumOfWeights = 0.f;
				for (int j = 0; j < 4; j++)
				{
					minWeightsPerVertex[j] = 1.f / (minDistances[j] * minDistances[j]);
					sumOfWeights += minWeightsPerVertex[j];
				}
				for (int j = 0; j < 4; j++)
				{
					minWeightsPerVertex[j] /= sumOfWeights;
					particleIndicesPerVertex.push_back(minParticleIndices[j]);
					weightsPerVertex.push_back(minWeightsPerVertex[j]);
					restPositionsPerVertex.push_back(minRestPositionsPerVertex[j]);
				}
			}
		}
	}

	m_deviceVertexData = std::make_shared<VertexData>();
	m_deviceVertexData->vertex = 0;
	m_deviceVertexData->restNormal = 0;
	m_deviceVertexData->particleIndex = 0;
	m_deviceVertexData->weight = 0;
	m_deviceVertexData->restPosition = 0;
	m_deviceVertexData->verticesCount = vertexPositions.size();

	cudaMalloc((void**)&m_deviceVertexData->vertex, 2 * vertexPositions.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceVertexData->restNormal, vertexNormals.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceVertexData->particleIndex, particleIndicesPerVertex.size() * sizeof(uint));
	cudaMalloc((void**)&m_deviceVertexData->weight, weightsPerVertex.size() * sizeof(float));
	cudaMalloc((void**)&m_deviceVertexData->restPosition, restPositionsPerVertex.size() * sizeof(float3));

	std::shared_ptr<VertexData> hostVertexData = std::make_shared<VertexData>();
	hostVertexData->restNormal = new float3[vertexNormals.size()];
	hostVertexData->particleIndex = new uint[particleIndicesPerVertex.size()];
	hostVertexData->weight = new float[weightsPerVertex.size()];
	hostVertexData->restPosition = new float3[restPositionsPerVertex.size()];

	for (int i = 0; i < vertexPositions.size(); i++)
	{
		hostVertexData->restNormal[i] = convertToFloat3(vertexNormals[i]);
		hostVertexData->particleIndex[4 * i + 0] = particleIndicesPerVertex[4 * i + 0];
		hostVertexData->particleIndex[4 * i + 1] = particleIndicesPerVertex[4 * i + 1];
		hostVertexData->particleIndex[4 * i + 2] = particleIndicesPerVertex[4 * i + 2];
		hostVertexData->particleIndex[4 * i + 3] = particleIndicesPerVertex[4 * i + 3];
		hostVertexData->weight[4 * i + 0] = weightsPerVertex[4 * i + 0];
		hostVertexData->weight[4 * i + 1] = weightsPerVertex[4 * i + 1];
		hostVertexData->weight[4 * i + 2] = weightsPerVertex[4 * i + 2];
		hostVertexData->weight[4 * i + 3] = weightsPerVertex[4 * i + 3];
		hostVertexData->restPosition[4 * i + 0] = convertToFloat3(restPositionsPerVertex[4 * i + 0]);
		hostVertexData->restPosition[4 * i + 1] = convertToFloat3(restPositionsPerVertex[4 * i + 1]);
		hostVertexData->restPosition[4 * i + 2] = convertToFloat3(restPositionsPerVertex[4 * i + 2]);
		hostVertexData->restPosition[4 * i + 3] = convertToFloat3(restPositionsPerVertex[4 * i + 3]);
	}

	cudaMemcpy(m_deviceVertexData->restNormal, hostVertexData->restNormal, vertexNormals.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceVertexData->particleIndex, hostVertexData->particleIndex, particleIndicesPerVertex.size() * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceVertexData->weight, hostVertexData->weight, weightsPerVertex.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceVertexData->restPosition, hostVertexData->restPosition, restPositionsPerVertex.size() * sizeof(float3), cudaMemcpyHostToDevice);
	
	m_vertexCount = vertexPositions.size();
	m_vertexIndexCount = vertexIndices.size();

	if (m_vertices)
		delete[] m_vertices;
	m_vertices = new float[m_vertexCount * 6];
	for (int i = 0; i < m_vertexCount; i++)
	{
		m_vertices[6 * i + 0] = transformedVertexPositions[i].x();
		m_vertices[6 * i + 1] = transformedVertexPositions[i].y();
		m_vertices[6 * i + 2] = transformedVertexPositions[i].z();
		m_vertices[6 * i + 3] = vertexNormals[i].x();
		m_vertices[6 * i + 4] = vertexNormals[i].y();
		m_vertices[6 * i + 5] = vertexNormals[i].z();
	}

	if (m_vertexIndices)
		delete[] m_vertexIndices;
	m_vertexIndices = new unsigned int[m_vertexIndexCount];
	for (int i = 0; i < m_vertexIndexCount; i++)
		m_vertexIndices[i] = vertexIndices[i];
}

void PBDSolver::prepareParticleBuffers(QVector<std::shared_ptr<ShapeConstraint>> shapes)
{
	// init and fill particle data
	m_deviceParticleData = std::make_shared<ParticleData>();
	m_deviceParticleData->position = 0;
	m_deviceParticleData->initialPosition = 0;
	m_deviceParticleData->predictedPosition = 0;
	m_deviceParticleData->velocity = 0;
	m_deviceParticleData->isPositionFixed = 0;
	m_deviceParticleData->mass = 0;
	m_deviceParticleData->collisionId = 0;
	m_deviceParticleData->gridSize = 64;
	m_deviceParticleData->gridCellIndex = 0;
	m_deviceParticleData->gridParticleIndex = 0;
	m_deviceParticleData->cellStart = 0;
	m_deviceParticleData->cellEnd = 0;
	m_deviceParticleData->shapeIndices = 0;
	m_deviceParticleData->endShapeIndex = 0;
	m_deviceParticleData->particleCount = m_particles.size();
	m_deviceParticleData->radius = 0.075f;
	m_deviceParticleData->raytestResult = 0;
	m_deviceParticleData->selectedParticleIndex = -1;

	for (auto particle : m_particles)
		particle->clearShapeIndices();

	for (int i = 0; i < shapes.size(); i++)
		for (auto particle : shapes[i]->getParticles())
			particle->addShapeIndex(i);

	int shapeIndicesCount = 0;
	for (auto particle : m_particles)
	{
		shapeIndicesCount += particle->getShapeIndices().size();
	}

	std::shared_ptr<ParticleData> hostParticleData = std::make_shared<ParticleData>();
	hostParticleData->position = new float3[m_particles.size()];
	hostParticleData->initialPosition = new float3[m_particles.size()];
	hostParticleData->velocity = new float3[m_particles.size()];
	hostParticleData->isPositionFixed = new bool[m_particles.size()];
	hostParticleData->mass = new float[m_particles.size()];
	hostParticleData->collisionId = new uint[m_particles.size()];
	hostParticleData->shapeIndices = new uint[shapeIndicesCount];
	hostParticleData->endShapeIndex = new uint[m_particles.size()];

	for (int i = 0, j = 0; i < m_particles.size(); i++)
	{
		hostParticleData->position[i] = convertToFloat3(m_particles[i]->getPosition());
		hostParticleData->initialPosition[i] = convertToFloat3(m_particles[i]->getInitialPosition());
		hostParticleData->velocity[i] = convertToFloat3(m_particles[i]->getVelocity());
		hostParticleData->isPositionFixed[i] = m_particles[i]->getIsPositionFixed();
		hostParticleData->mass[i] = m_particles[i]->getMass();
		hostParticleData->collisionId[i] = m_particles[i]->getCollisionId();

		for (int shapeIndex : m_particles[i]->getShapeIndices())
		{
			hostParticleData->shapeIndices[j] = shapeIndex;
			j++;
		}

		hostParticleData->endShapeIndex[i] = j - 1;
	}

	if (m_particlePositions)
		delete[] m_particlePositions;
	m_particlePositions = new float[m_particles.size() * 3];
	for (int i = 0; i < m_particles.size(); i++)
	{
		m_particlePositions[3 * i + 0] = m_particles[i]->getPosition().x();
		m_particlePositions[3 * i + 1] = m_particles[i]->getPosition().y();
		m_particlePositions[3 * i + 2] = m_particles[i]->getPosition().z();
	}

	cudaMalloc((void**)&m_deviceParticleData->position, m_particles.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceParticleData->initialPosition, m_particles.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceParticleData->predictedPosition, m_particles.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceParticleData->velocity, m_particles.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceParticleData->isPositionFixed, m_particles.size() * sizeof(bool));
	cudaMalloc((void**)&m_deviceParticleData->mass, m_particles.size() * sizeof(float));
	cudaMalloc((void**)&m_deviceParticleData->collisionId, m_particles.size() * sizeof(uint));
	cudaMalloc((void**)&m_deviceParticleData->gridCellIndex, m_particles.size() * sizeof(uint));
	cudaMalloc((void**)&m_deviceParticleData->gridParticleIndex, m_particles.size() * sizeof(uint));
	cudaMalloc((void**)&m_deviceParticleData->cellStart, pow(m_deviceParticleData->gridSize, 3) * sizeof(int));
	cudaMalloc((void**)&m_deviceParticleData->cellEnd, pow(m_deviceParticleData->gridSize, 3) * sizeof(int));
	cudaMalloc((void**)&m_deviceParticleData->shapeIndices, shapeIndicesCount * sizeof(uint));
	cudaMalloc((void**)&m_deviceParticleData->endShapeIndex, m_particles.size() * sizeof(uint));
	cudaMalloc((void**)&m_deviceParticleData->raytestResult, m_particles.size() * sizeof(float));

	cudaMemcpy(m_deviceParticleData->position, hostParticleData->position, m_particles.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->initialPosition, hostParticleData->initialPosition, m_particles.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->velocity, hostParticleData->velocity, m_particles.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->isPositionFixed, hostParticleData->isPositionFixed, m_particles.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->mass, hostParticleData->mass, m_particles.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->collisionId, hostParticleData->collisionId, m_particles.size() * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->shapeIndices, hostParticleData->shapeIndices, shapeIndicesCount * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceParticleData->endShapeIndex, hostParticleData->endShapeIndex, m_particles.size() * sizeof(uint), cudaMemcpyHostToDevice);
}

void PBDSolver::prepareShapeBuffers(QVector<std::shared_ptr<ShapeConstraint>> shapes)
{
	// init and fill shape data
	m_deviceShapeData = std::make_shared<ShapeData>();
	m_deviceShapeData->Aqq = 0;
	m_deviceShapeData->AqqTilde = 0;
	m_deviceShapeData->currentRotation = 0;
	m_deviceShapeData->initialCenterOfMass = 0;
	m_deviceShapeData->currentCenterOfMass = 0;
	m_deviceShapeData->finalDeformation = 0;
	m_deviceShapeData->finalDeformationTilde = 0;
	m_deviceShapeData->type = 0;
	m_deviceShapeData->particleIndices = 0;
	m_deviceShapeData->endParticleIndex = 0;
	m_deviceShapeData->shapeCount = shapes.size();

	for (auto shape : shapes)
		shape->clearParticleIndices();

	for (int i = 0; i < m_particles.size(); i++)
		for (int shapeIndex : m_particles[i]->getShapeIndices())
			shapes[shapeIndex]->addParticleIndex(i);

	int particleIndicesCount = 0;
	for (auto shape : shapes)
	{
		particleIndicesCount += shape->getParticleIndices().size();
	}

	std::shared_ptr<ShapeData> hostShapeData = std::make_shared<ShapeData>();
	hostShapeData->Aqq = new float3x3[shapes.size()];
	hostShapeData->AqqTilde = new float9x9[shapes.size()];
	hostShapeData->type = new uint[shapes.size()];
	hostShapeData->currentRotation = new float3x3[shapes.size()];
	hostShapeData->initialCenterOfMass = new float3[shapes.size()];
	hostShapeData->particleIndices = new uint[particleIndicesCount];
	hostShapeData->endParticleIndex = new uint[shapes.size()];

	for (int i = 0, j = 0; i < shapes.size(); i++)
	{
		hostShapeData->Aqq[i] = convertToFloat3x3(shapes[i]->getAqq());
		hostShapeData->currentRotation[i] = convertToFloat3x3(shapes[i]->getRotation());
		hostShapeData->initialCenterOfMass[i] = convertToFloat3(shapes[i]->getInitialCenterOfMass());

		if (std::shared_ptr<RigidShapeConstraint> rigidShape = std::dynamic_pointer_cast<RigidShapeConstraint>(shapes[i]))
			hostShapeData->type[i] = 0;
		else if (std::shared_ptr<SoftShapeLinearConstraint> softLinearShape = std::dynamic_pointer_cast<SoftShapeLinearConstraint>(shapes[i]))
			hostShapeData->type[i] = 1;
		else if (std::shared_ptr<SoftShapeQuadraticConstraint> softQuadraticShape = std::dynamic_pointer_cast<SoftShapeQuadraticConstraint>(shapes[i]))
		{
			hostShapeData->type[i] = 2;
			hostShapeData->AqqTilde[i] = convertToFloat9x9(softQuadraticShape->getAqqTilde());
		}

		for (int particleIndex : shapes[i]->getParticleIndices())
		{
			hostShapeData->particleIndices[j] = particleIndex;
			j++;
		}

		hostShapeData->endParticleIndex[i] = j - 1;
	}

	cudaMalloc((void**)&m_deviceShapeData->Aqq, shapes.size() * sizeof(float3x3));
	cudaMalloc((void**)&m_deviceShapeData->AqqTilde, shapes.size() * sizeof(float9x9));
	cudaMalloc((void**)&m_deviceShapeData->currentRotation, shapes.size() * sizeof(float3x3));
	cudaMalloc((void**)&m_deviceShapeData->initialCenterOfMass, shapes.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceShapeData->currentCenterOfMass, shapes.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceShapeData->finalDeformation, shapes.size() * sizeof(float3x3));
	cudaMalloc((void**)&m_deviceShapeData->finalDeformationTilde, shapes.size() * sizeof(float3x9));
	cudaMalloc((void**)&m_deviceShapeData->type, shapes.size() * sizeof(uint));
	cudaMalloc((void**)&m_deviceShapeData->particleIndices, particleIndicesCount * sizeof(uint));
	cudaMalloc((void**)&m_deviceShapeData->endParticleIndex, shapes.size() * sizeof(uint));

	cudaMemcpy(m_deviceShapeData->Aqq, hostShapeData->Aqq, shapes.size() * sizeof(float3x3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceShapeData->AqqTilde, hostShapeData->AqqTilde, shapes.size() * sizeof(float9x9), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceShapeData->currentRotation, hostShapeData->currentRotation, shapes.size() * sizeof(float3x3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceShapeData->initialCenterOfMass, hostShapeData->initialCenterOfMass, shapes.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceShapeData->type, hostShapeData->type, shapes.size() * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceShapeData->particleIndices, hostShapeData->particleIndices, particleIndicesCount * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceShapeData->endParticleIndex, hostShapeData->endParticleIndex, shapes.size() * sizeof(uint), cudaMemcpyHostToDevice);
}

void PBDSolver::prepareStaticCollisionBuffers()
{
	// plane constraints
	QVector<std::shared_ptr<PlaneConstraint>> planes;
	for (auto constraint : m_constraints)
		if (std::shared_ptr<PlaneConstraint> planeConstraint = std::dynamic_pointer_cast<PlaneConstraint>(constraint))
			planes.push_back(planeConstraint);

	// init static collision plane data
	m_deviceColliderData = std::make_shared<ColliderData>();
	m_deviceColliderData->planeCount = planes.size();
	m_deviceColliderData->planeNormal = 0;
	m_deviceColliderData->planeDistance = 0;

	cudaMalloc((void**)&m_deviceColliderData->planeNormal, planes.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceColliderData->planeDistance, planes.size() * sizeof(float));

	// box constraints
	QVector<std::shared_ptr<BoxConstraint>> boxes;
	for (auto constraint : m_constraints)
		if (std::shared_ptr<BoxConstraint> boxConstraint = std::dynamic_pointer_cast<BoxConstraint>(constraint))
			boxes.push_back(boxConstraint);

	// init static collision plane data
	m_deviceColliderData->boxCount = boxes.size();
	m_deviceColliderData->boxPosition = 0;
	m_deviceColliderData->boxHalfDimension = 0;
	m_deviceColliderData->boxRotation = 0;
	m_deviceColliderData->boxIsBoundary = 0;

	cudaMalloc((void**)&m_deviceColliderData->boxPosition, boxes.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceColliderData->boxHalfDimension, boxes.size() * sizeof(float3));
	cudaMalloc((void**)&m_deviceColliderData->boxRotation, boxes.size() * sizeof(float3x3));
	cudaMalloc((void**)&m_deviceColliderData->boxIsBoundary, boxes.size() * sizeof(bool));

	updateStaticCollisionGPUBuffers();
}

void PBDSolver::updateStaticCollisionGPUBuffers()
{
	// plane constraints
	QVector<std::shared_ptr<PlaneConstraint>> planes;
	for (auto constraint : m_constraints)
		if (std::shared_ptr<PlaneConstraint> planeConstraint = std::dynamic_pointer_cast<PlaneConstraint>(constraint))
			planes.push_back(planeConstraint);

	std::shared_ptr<ColliderData> hostColliderData = std::make_shared<ColliderData>();
	hostColliderData->planeNormal = new float3[planes.size()];
	hostColliderData->planeDistance = new float[planes.size()];

	for (int i = 0; i < planes.size(); i++)
	{
		hostColliderData->planeNormal[i] = convertToFloat3(planes[i]->getNormal());
		hostColliderData->planeDistance[i] = planes[i]->getDistance();
	}

	cudaMemcpy(m_deviceColliderData->planeNormal, hostColliderData->planeNormal, planes.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceColliderData->planeDistance, hostColliderData->planeDistance, planes.size() * sizeof(float), cudaMemcpyHostToDevice);

	// box constraints
	QVector<std::shared_ptr<BoxConstraint>> boxes;
	for (auto constraint : m_constraints)
		if (std::shared_ptr<BoxConstraint> boxConstraint = std::dynamic_pointer_cast<BoxConstraint>(constraint))
			boxes.push_back(boxConstraint);

	hostColliderData->boxPosition = new float3[boxes.size()];
	hostColliderData->boxHalfDimension = new float3[boxes.size()];
	hostColliderData->boxRotation = new float3x3[boxes.size()];
	hostColliderData->boxIsBoundary = new bool[boxes.size()];

	for (int i = 0; i < boxes.size(); i++)
	{
		hostColliderData->boxPosition[i] = convertToFloat3(boxes[i]->getPosition());
		hostColliderData->boxHalfDimension[i] = convertToFloat3(boxes[i]->getHalfDimension());
		hostColliderData->boxRotation[i] = convertToFloat3x3(boxes[i]->getRotation());
		hostColliderData->boxIsBoundary[i] = boxes[i]->getIsBoundary();
	}

	cudaMemcpy(m_deviceColliderData->boxPosition, hostColliderData->boxPosition, boxes.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceColliderData->boxHalfDimension, hostColliderData->boxHalfDimension, boxes.size() * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceColliderData->boxRotation, hostColliderData->boxRotation, boxes.size() * sizeof(float3x3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_deviceColliderData->boxIsBoundary, hostColliderData->boxIsBoundary, boxes.size() * sizeof(bool), cudaMemcpyHostToDevice);
}