#include "CUDA/CorrectParticles.cuh"

void correctParticlesWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData, float deltaTime)
{
	correctParticlesKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData, parameterData, deltaTime);
}

__global__ void correctParticlesKernel(ParticleData particleData, PBDParameterData parameterData, float deltaTime)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	if (!particleData.isPositionFixed[particleIndex])
	{
		float3 predictedPostion = particleData.predictedPosition[particleIndex];
		float3 deltaPosition = (predictedPostion - particleData.position[particleIndex]);
		particleData.velocity[particleIndex] = deltaPosition / deltaTime;

		if (length(deltaPosition) > parameterData.particleSleepingThreshold)
		{
			particleData.position[particleIndex] = predictedPostion;
		}
	}
}