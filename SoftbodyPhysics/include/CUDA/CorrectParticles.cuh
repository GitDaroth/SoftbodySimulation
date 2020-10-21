#pragma once

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA/ParticleData.h"
#include "PBDParameterData.h"

void correctParticlesWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData, float deltaTime);
__global__ void correctParticlesKernel(ParticleData particleData, PBDParameterData parameterData, float deltaTime);