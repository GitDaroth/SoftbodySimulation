#pragma once

#include "CUDAMath.h"

struct ShapeData
{
	float3* initialCenterOfMass;		// size: #clusters
	float3* currentCenterOfMass;		// size: #clusters
	float3x3* currentRotation;			// size: #clusters
	float3x3* Aqq;						// size: #clusters
	float9x9* AqqTilde;					// size: #clusters
	float3x3* finalDeformation;			// size: #clusters
	float3x9* finalDeformationTilde;	// size: #clusters
	uint* type;							// size: #clusters (0 = rigid, 1 = soft linear, 2 = soft quadratic)
	uint* particleIndices;				// size: sum of #particles in clusters
	uint* endParticleIndex;				// size: #clusters
	uint shapeCount;
};