#include "CUDA/SkinMesh.cuh"

void skinMeshWithCuda(std::shared_ptr<VertexData> vertexData, std::shared_ptr<ParticleData> particleData, std::shared_ptr<ShapeData> shapeData)
{
	skinMeshKernel<<<(vertexData->verticesCount + 511) / 512, 512 >>>(*vertexData, *particleData, *shapeData);
}

__global__ void skinMeshKernel(VertexData vertexData, ParticleData particleData, ShapeData shapeData)
{
	int vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (vertexIndex >= vertexData.verticesCount)
		return;

	float selectedParticleWeight = 0.f;
	for (int i = 0; i < 4; i++)
	{
		uint particleIndex = vertexData.particleIndex[4 * vertexIndex + i];
		if (particleData.isPositionFixed[particleIndex])
			selectedParticleWeight = vertexData.weight[4 * vertexIndex + i];
	}

	float3 skinnedVertexPosition = make_float3(0.f);
	float3 skinnedVertexNormal = make_float3(0.f);
	for (int i = 0; i < 4; i++)
	{
		uint particleIndex = vertexData.particleIndex[4 * vertexIndex + i];
		if (!particleData.isPositionFixed[particleIndex])
		{
			float3x3 rotation = shapeData.currentRotation[particleData.shapeIndices[particleData.endShapeIndex[particleIndex]]];
			float weight = selectedParticleWeight / 3.f + vertexData.weight[4 * vertexIndex + i];
			skinnedVertexPosition += weight * (rotation * vertexData.restPosition[4 * vertexIndex + i] + particleData.position[particleIndex]);
			skinnedVertexNormal += weight * (rotation * vertexData.restNormal[vertexIndex]);
		}
	}

	vertexData.vertex[2 * vertexIndex] = skinnedVertexPosition;
	vertexData.vertex[2 * vertexIndex + 1] = skinnedVertexNormal;
}