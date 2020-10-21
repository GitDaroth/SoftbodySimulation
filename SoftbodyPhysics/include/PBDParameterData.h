#pragma once

#include <CUDAMath.h>

struct PBDParameterData
{
	// general parameters
	uint solverIterations;
	float3 gravity;
	float damping;
	float particleSleepingThreshold;

	// shape parameters
	uint rotExtractionIterations;
	float deformationThreshold;
	float stiffness_softLinear;
	float stiffness_softQuadratic;
	float beta_softLinear;
	float beta_softQuadratic;
	bool useVolumeConservation_softLinear;

	// collision parameters
	float staticFriction;
	float dynamicFriction;
};