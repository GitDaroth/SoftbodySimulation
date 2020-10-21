#include "CUDA/ConstructGrid.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

void constructGridWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData)
{
	calcGridCellsKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData, parameterData);

	// sort by cell indices
	auto gridCellIndexStart = thrust::device_pointer_cast(particleData->gridCellIndex);
	auto gridCellIndexEnd = thrust::device_pointer_cast(particleData->gridCellIndex + particleData->particleCount);
	auto gridParticleIndex = thrust::device_pointer_cast(particleData->gridParticleIndex);

	auto stream = thrust::cuda::par.on(0);
	thrust::sort_by_key(stream, gridCellIndexStart, gridCellIndexEnd, gridParticleIndex);

	// fill cell start and end with -1
	auto arrayStart = thrust::device_pointer_cast(particleData->cellStart);
	auto arrayEnd = thrust::device_pointer_cast(particleData->cellStart + (uint)pow(particleData->gridSize, 3));
	thrust::fill(stream, arrayStart, arrayEnd, -1);

	arrayStart = thrust::device_pointer_cast(particleData->cellEnd);
	arrayEnd = thrust::device_pointer_cast(particleData->cellEnd + (uint)pow(particleData->gridSize, 3));
	thrust::fill(stream, arrayStart, arrayEnd, -1);

	calcCellStartAndEndKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData);
}

__global__ void calcGridCellsKernel(ParticleData particleData, PBDParameterData parameterData)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	// calc cell pos
	float3 position = particleData.predictedPosition[particleIndex];
	float cellSize = 2.f * particleData.radius;
	float3 worldSizeHalf = make_float3(10.f);
	int3 cellPos;
	cellPos.x = (int)((position.x + worldSizeHalf.x) / cellSize);
	cellPos.y = (int)((position.y + worldSizeHalf.y) / cellSize);
	cellPos.z = (int)((position.z + worldSizeHalf.z) / cellSize);

	// calc cell hash
	// only allow cell positions within gridSize
	uint x = cellPos.x & (particleData.gridSize - 1);
	uint y = cellPos.y & (particleData.gridSize - 1);
	uint z = cellPos.z & (particleData.gridSize - 1);
	uint cellHash = z * particleData.gridSize * particleData.gridSize + y * particleData.gridSize + x;

	particleData.gridCellIndex[particleIndex] = cellHash;
	particleData.gridParticleIndex[particleIndex] = particleIndex;
}

__global__ void calcCellStartAndEndKernel(ParticleData particleData)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	uint currentCellIndex = particleData.gridCellIndex[particleIndex];
	if (particleIndex == 0)
	{
		particleData.cellStart[currentCellIndex] = particleIndex;
		return;
	}

	if (particleIndex == particleData.particleCount - 1)
	{
		particleData.cellEnd[currentCellIndex] = particleIndex;
		return;
	}

	uint nextCellIndex = particleData.gridCellIndex[particleIndex + 1];
	if (currentCellIndex < nextCellIndex)
	{
		particleData.cellEnd[currentCellIndex] = particleIndex;
		particleData.cellStart[nextCellIndex] = particleIndex + 1;
	}
}