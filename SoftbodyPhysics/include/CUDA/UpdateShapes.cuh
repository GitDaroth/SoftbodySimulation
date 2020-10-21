#pragma once

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA/ParticleData.h"
#include "CUDA/ShapeData.h"
#include "PBDParameterData.h"

void updateShapesWithCuda(std::shared_ptr<ShapeData> shapeData, std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData);
__global__ void updateShapesKernel(ShapeData shapeData, ParticleData particleData, PBDParameterData parameterData);