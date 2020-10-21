#include "CUDA/SolveConstraints.cuh"

void solveConstraintsWithCuda(std::shared_ptr<ParticleData> particleData, std::shared_ptr<ShapeData> shapeData, std::shared_ptr<ColliderData> colliderData, PBDParameterData parameterData)
{
	solveConstraintsKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData, *shapeData, *colliderData, parameterData);
}

void solveParticleCollisionsWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData)
{
	solveParticleCollisionKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData, parameterData);
}

__global__ void solveConstraintsKernel(ParticleData particleData, ShapeData shapeData, ColliderData colliderData, PBDParameterData parameterData)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	if (!particleData.isPositionFixed[particleIndex])
	{
		float3 predictedPosition = particleData.predictedPosition[particleIndex];

		// solve shape constraints
		// start and end shape index for a particle
		uint startShapeIndex = 0;
		if (particleIndex > 0)
			startShapeIndex = particleData.endShapeIndex[particleIndex - 1] + 1;
		uint endShapeIndex = particleData.endShapeIndex[particleIndex];

		for (uint i = startShapeIndex; i <= endShapeIndex; i++)
		{
			uint shapeIndex = particleData.shapeIndices[i];
			uint shapeType = shapeData.type[shapeIndex];

			float3 q = particleData.initialPosition[particleIndex] - shapeData.initialCenterOfMass[shapeIndex];
			float3 goalPosition;
			float stiffness;
			if (shapeType == 0)
			{
				stiffness = 1.f;
				goalPosition = shapeData.finalDeformation[shapeIndex] * q + shapeData.currentCenterOfMass[shapeIndex];
			}
			else if (shapeType == 1)
			{
				stiffness = parameterData.stiffness_softLinear;
				goalPosition = shapeData.finalDeformation[shapeIndex] * q + shapeData.currentCenterOfMass[shapeIndex];
			}
			else if (shapeType == 2)
			{
				stiffness = parameterData.stiffness_softQuadratic;
				float9 qTilde = make_float9(q.x, q.y, q.z, q.x * q.x, q.y * q.y, q.z * q.z, q.x * q.y, q.y * q.z, q.z * q.x);
				goalPosition = shapeData.finalDeformationTilde[shapeIndex] * qTilde + shapeData.currentCenterOfMass[shapeIndex];
			}
			float3 deltaPosition = goalPosition - predictedPosition;
			float linearlyDependentStiffness = 1.0f - powf(1.f - stiffness, 1.f / ((float)parameterData.solverIterations));
			predictedPosition += deltaPosition * linearlyDependentStiffness;
		}

		// solve plane constraints
		for (int i = 0; i < colliderData.planeCount; i++)
		{
			float3 normal = colliderData.planeNormal[i];
			float d = colliderData.planeDistance[i];
			float penetration = d - dot(normal, predictedPosition) + particleData.radius;
			if (penetration > 0.f)
			{
				float3 velocity = particleData.predictedPosition[particleIndex] - particleData.position[particleIndex];
				float3 deltaPosition = calcCollisionResponse(normal, velocity, penetration, parameterData);
				predictedPosition += deltaPosition;
			}
		}

		// solve box constraints
		for (int i = 0; i < colliderData.boxCount; i++)
		{
			float3x3 boxRotation = colliderData.boxRotation[i];
			float3 boxHalfDimension = colliderData.boxHalfDimension[i];
			float3 boxPosition = colliderData.boxPosition[i];

			float3 distance = transpose(boxRotation) * (predictedPosition - boxPosition);

			if (colliderData.boxIsBoundary[i]) // collision boundary
			{
				float3 penetrationVec = abs(distance) - boxHalfDimension + make_float3(particleData.radius);
				float3 velocity = particleData.predictedPosition[particleIndex] - particleData.position[particleIndex];
				if (penetrationVec.x > 0.f || penetrationVec.y > 0.f || penetrationVec.z > 0.f)
				{
					if (penetrationVec.x > 0.f)
					{
						float3 normal = make_float3(0.f);
						if (distance.x > 0.f)
							normal.x = -1.f;
						else
							normal.x = 1.f;
#
						float3 deltaPosition = calcCollisionResponse(boxRotation * normal, velocity, penetrationVec.x, parameterData);
						predictedPosition += deltaPosition;
					}
					if (penetrationVec.y > 0.f)
					{
						float3 normal = make_float3(0.f);
						if (distance.y > 0.f)
							normal.y = -1.f;
						else
							normal.y = 1.f;

						float3 deltaPosition = calcCollisionResponse(boxRotation * normal, velocity, penetrationVec.y, parameterData);
						predictedPosition += deltaPosition;
					}
					if (penetrationVec.z > 0.f)
					{
						float3 normal = make_float3(0.f);
						if (distance.z > 0.f)
							normal.z = -1.f;
						else
							normal.z = 1.f;

						float3 deltaPosition = calcCollisionResponse(boxRotation * normal, velocity, penetrationVec.z, parameterData);
						predictedPosition += deltaPosition;
					}
				}
			}
			else // collision object
			{
				float3 normal = make_float3(0.f);
				float3 penetrationVec = boxHalfDimension - abs(distance) + make_float3(particleData.radius);
				float penetration = 0.f;
				if (penetrationVec.x > 0.f && penetrationVec.y > 0.f && penetrationVec.z > 0.f)
				{
					// Check if penetration on X-Axis is the smallest
					if (penetrationVec.x < penetrationVec.y && penetrationVec.x < penetrationVec.z)
					{
						penetration = penetrationVec.x;
						if (distance.x < 0.f)
							normal.x = -1.f;
						else
							normal.x = 1.f;
					}
					// Check if penetration on Y-Axis is the smallest
					else if (penetrationVec.y < penetrationVec.z && penetrationVec.y < penetrationVec.x)
					{
						penetration = penetrationVec.y;
						if (distance.y < 0.f)
							normal.y = -1.f;
						else
							normal.y = 1.f;
					}
					// Check if penetration on Z-Axis is the smallest
					else if (penetrationVec.z < penetrationVec.x && penetrationVec.z < penetrationVec.y)
					{
						penetration = penetrationVec.z;
						if (distance.z < 0.f)
							normal.z = -1.f;
						else
							normal.z = 1.f;
					}

					float3 velocity = particleData.predictedPosition[particleIndex] - particleData.position[particleIndex];
					float3 deltaPosition = calcCollisionResponse(boxRotation * normal, velocity, penetration, parameterData);
					predictedPosition += deltaPosition;
				}
			}
		}

		particleData.predictedPosition[particleIndex] = predictedPosition;
	}
}

__global__ void solveParticleCollisionKernel(ParticleData particleData, PBDParameterData parameterData)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	if (!particleData.isPositionFixed[particleIndex])
	{
		// solve particle collision constraints
		//// BRUTE FORCE
		//for (int i = 0; i < particleData.particleCount; i++)
		//{
		//	if (i == particleIndex || particleData.collisionId[i] == particleData.collisionId[particleIndex])
		//		continue;

		//	float3 relativePosition = particleData.predictedPosition[particleIndex] - particleData.predictedPosition[i];
		//	float distance = length(relativePosition);
		//	if (distance < 2.f * particleData.radius)
		//	{
		//		float3 deltaPosition = -0.5f * (distance - 2.f * particleData.radius) * relativePosition / distance;
		//		particleData.predictedPosition[particleIndex] += deltaPosition;
		//		particleData.predictedPosition[i] -= deltaPosition;
		//	}
		//}

		// WITH NEIGHBOR SEARCH
		float3 position = particleData.predictedPosition[particleIndex];
		float cellSize = 2.f * particleData.radius;
		float3 worldSizeHalf = make_float3(10.f);
		int3 cellPos;
		cellPos.x = (int)((position.x + worldSizeHalf.x) / cellSize);
		cellPos.y = (int)((position.y + worldSizeHalf.y) / cellSize);
		cellPos.z = (int)((position.z + worldSizeHalf.z) / cellSize);
		// iterate through neighboring cells
		for (int x = cellPos.x - 1; x <= cellPos.x + 1; x++)
		{
			for (int y = cellPos.y - 1; y <= cellPos.y + 1; y++)
			{
				for (int z = cellPos.z - 1; z <= cellPos.z + 1; z++)
				{
					uint cellX = x & (particleData.gridSize - 1);
					uint cellY = y & (particleData.gridSize - 1);
					uint cellZ = z & (particleData.gridSize - 1);
					uint cellHash = cellZ * particleData.gridSize * particleData.gridSize + cellY * particleData.gridSize + cellX;

					int cellStart = particleData.cellStart[cellHash];
					int cellEnd = particleData.cellEnd[cellHash];
					if (cellStart < 0 || cellEnd < 0)
						continue;

					// iterate through particles in cell
					for (int i = cellStart; i <= cellEnd; i++)
					{
						int neighborParticleIndex = particleData.gridParticleIndex[i];
						if (neighborParticleIndex == particleIndex || particleData.collisionId[neighborParticleIndex] == particleData.collisionId[particleIndex])
							continue;

						float3 relativePosition = particleData.predictedPosition[particleIndex] - particleData.predictedPosition[neighborParticleIndex];
						float distance = length(relativePosition);
						if (distance < 2.f * particleData.radius)
						{
							float3 deltaPosition = -0.5f * (distance - 2.f * particleData.radius) * relativePosition / distance;
							particleData.predictedPosition[particleIndex] += deltaPosition;
							particleData.predictedPosition[neighborParticleIndex] -= deltaPosition;

							float penetration = distance - 2.f * particleData.radius;
							float3 normal = relativePosition / distance;
							float3 relativeVelocity = (particleData.predictedPosition[particleIndex] - particleData.position[particleIndex]) - (particleData.predictedPosition[neighborParticleIndex] - particleData.position[neighborParticleIndex]);
							float3 tangentialDisplacement = relativeVelocity - dot(relativeVelocity, normal) * normal;

							if (length(tangentialDisplacement) < parameterData.staticFriction * penetration)
								deltaPosition = 0.5f * tangentialDisplacement;
							else if (length(tangentialDisplacement) != 0.f)
								deltaPosition = 0.5f * tangentialDisplacement * fminf((parameterData.dynamicFriction * penetration) / length(tangentialDisplacement), 1.f);

							particleData.predictedPosition[particleIndex] += deltaPosition;
							particleData.predictedPosition[neighborParticleIndex] -= deltaPosition;
						}
					}
				}
			}
		}
	}
}

__device__ float3 calcCollisionResponse(float3 normal, float3 velocity, float penetration, PBDParameterData parameterData)
{
	float3 collisionResponse = penetration * normal;
	float3 tangentialDisplacement = velocity - dot(velocity, normal) * normal;

	if (length(tangentialDisplacement) < parameterData.staticFriction * -penetration)
		collisionResponse += 0.5f * tangentialDisplacement;
	else if (length(tangentialDisplacement) != 0.f)
		collisionResponse += 0.5f * tangentialDisplacement * fminf((parameterData.dynamicFriction * -penetration) / length(tangentialDisplacement), 1.f);

	return collisionResponse;
}