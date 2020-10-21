#pragma once

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA/ParticleData.h"
#include "CUDA/ShapeData.h"
#include "CUDA/ColliderData.h"
#include "PBDParameterData.h"

void solveConstraintsWithCuda(std::shared_ptr<ParticleData> particleData, std::shared_ptr<ShapeData> shapeData, std::shared_ptr<ColliderData> colliderData, PBDParameterData parameterData);
void solveParticleCollisionsWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData);
__global__ void solveConstraintsKernel(ParticleData particleData, ShapeData shapeData, ColliderData colliderData, PBDParameterData parameterData);
__global__ void solveParticleCollisionKernel(ParticleData particleData, PBDParameterData parameterData);
__device__ float3 calcCollisionResponse(float3 normal, float3 velocity, float penetration, PBDParameterData parameterData);
