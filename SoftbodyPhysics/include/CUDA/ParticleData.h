#pragma once

#include "CUDAMath.h"

struct ParticleData
{
	float3* position;				// size: #particles
	float3* initialPosition;		// size: #particles
	float3* predictedPosition;		// size: #particles
	float3* velocity;				// size: #particles
	bool* isPositionFixed;			// size: #particles
	float* mass;					// size: #particles
	uint* collisionId;				// size: #particles
	uint gridSize;
	uint* gridCellIndex;			// size: #particles
	uint* gridParticleIndex;		// size: #particles
	int* cellStart;					// size: #cells
	int* cellEnd;					// size: #cells
	uint* shapeIndices;				// size: sum of #clusters in particles
	uint* endShapeIndex;			// size: #particles
	uint particleCount;
	float radius;

	float* raytestResult;			// size: #particles
	int selectedParticleIndex;
	float selectedParticleDistance;
	float3 selectedParticlePosition;
};