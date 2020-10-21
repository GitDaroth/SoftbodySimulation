#include "CUDA/RaytestParticles.cuh"

void raytestParticlesWithCuda(std::shared_ptr<ParticleData> particleData, float3 rayOrigin, float3 rayDircetion)
{
	raytestParticlesKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData, rayOrigin, rayDircetion);
}

__global__ void raytestParticlesKernel(ParticleData particleData, float3 rayOrigin, float3 rayDircetion)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	float3 distanceVec = rayOrigin - particleData.predictedPosition[particleIndex];
	float b = dot(rayDircetion, distanceVec);
	float c = dot(distanceVec, distanceVec) - particleData.radius * particleData.radius;

	if (b * b - c < 0)
		particleData.raytestResult[particleIndex] = -1.0f;
	else
		particleData.raytestResult[particleIndex] = -b + sqrtf(b * b - c);
}