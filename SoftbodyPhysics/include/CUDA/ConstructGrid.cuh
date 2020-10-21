#pragma once

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA/ParticleData.h"
#include "PBDParameterData.h"

void constructGridWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData);
__global__ void calcGridCellsKernel(ParticleData particleData, PBDParameterData parameterData);
__global__ void calcCellStartAndEndKernel(ParticleData particleData);