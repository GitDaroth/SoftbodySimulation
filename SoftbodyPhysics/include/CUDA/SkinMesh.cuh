#pragma once

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA/ParticleData.h"
#include "CUDA/ShapeData.h"
#include "CUDA/VertexData.h"

void skinMeshWithCuda(std::shared_ptr<VertexData> vertexData, std::shared_ptr<ParticleData> particleData, std::shared_ptr<ShapeData> shapeData);
__global__ void skinMeshKernel(VertexData vertexData, ParticleData particleData, ShapeData shapeData);