#pragma once

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA/ParticleData.h"

void raytestParticlesWithCuda(std::shared_ptr<ParticleData> particleData, float3 rayOrigin, float3 rayDircetion);
__global__ void raytestParticlesKernel(ParticleData particleData, float3 rayOrigin, float3 rayDircetion);