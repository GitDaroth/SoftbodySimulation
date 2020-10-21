#pragma once

#include "CUDAMath.h"

struct ColliderData
{
	uint planeCount;
	float3* planeNormal;	// size: #planes
	float* planeDistance;	// size: #planes

	uint boxCount;
	float3* boxPosition;		// size: #boxes
	float3* boxHalfDimension;	// size: #boxes
	float3x3* boxRotation;		// size: #boxes
	bool* boxIsBoundary;		// size: #boxes
};