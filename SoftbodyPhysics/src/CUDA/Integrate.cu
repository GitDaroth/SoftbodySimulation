#include "CUDA/Integrate.cuh"

void integrateWithCuda(std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData, float deltaTime)
{
	integrateKernel<<<(particleData->particleCount + 511) / 512, 512>>>(*particleData, parameterData, deltaTime);
}

__global__ void integrateKernel(ParticleData particleData, PBDParameterData parameterData, float deltaTime)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particleData.particleCount)
		return;

	if (particleData.selectedParticleIndex < 0)
	{
		particleData.isPositionFixed[particleIndex] = false;
	}
	else
	{
		particleData.isPositionFixed[particleData.selectedParticleIndex] = true;
		particleData.predictedPosition[particleData.selectedParticleIndex] = particleData.selectedParticlePosition;
		particleData.position[particleData.selectedParticleIndex] = particleData.selectedParticlePosition;
	}


	if (!particleData.isPositionFixed[particleIndex])
	{
		float3 velocityTemp = particleData.velocity[particleIndex];
		velocityTemp += deltaTime * parameterData.gravity;
		velocityTemp = velocityTemp * parameterData.damping;

		particleData.predictedPosition[particleIndex] = particleData.position[particleIndex] + deltaTime * velocityTemp;
	}
}