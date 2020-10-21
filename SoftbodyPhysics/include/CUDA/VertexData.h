#pragma once

#include "CUDAMath.h"

struct VertexData
{
	float3* vertex;			// size: 2 * #vertices (float3 position, float3 normal)
	float3* restNormal;		// size: #vertices
	uint* particleIndex;	// size: 4 * #vertices
	float* weight;			// size: 4 * #vertices
	float3* restPosition;	// size: 4 * #vertices
	uint verticesCount;
};