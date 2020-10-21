#include "CUDA/UpdateShapes.cuh"

void updateShapesWithCuda(std::shared_ptr<ShapeData> shapeData, std::shared_ptr<ParticleData> particleData, PBDParameterData parameterData)
{
	updateShapesKernel<<<(shapeData->shapeCount + 127) / 128, 128 >>>(*shapeData, *particleData, parameterData);
}

__global__ void updateShapesKernel(ShapeData shapeData, ParticleData particleData, PBDParameterData parameterData)
{
	int shapeIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (shapeIndex >= shapeData.shapeCount)
		return;

	// start and end particle index for a shape
	uint startParticleIndex = 0;
	if (shapeIndex > 0)
		startParticleIndex = shapeData.endParticleIndex[shapeIndex - 1] + 1;
	uint endParticleIndex = shapeData.endParticleIndex[shapeIndex];

	// calculate current center of mass
	float totalMass = 0.0;
	float3 weightedPosition = make_float3(0.0);
	for (uint i = startParticleIndex; i <= endParticleIndex; i++)
	{
		uint particleIndex = shapeData.particleIndices[i];
		float mass = particleData.mass[particleIndex];

		totalMass += mass;
		weightedPosition += mass * particleData.predictedPosition[particleIndex];
	}

	float3 centerOfMass = weightedPosition / totalMass;
	shapeData.currentCenterOfMass[shapeIndex] = centerOfMass;

	// Apq ausrechnen
	float3 initialCenterOfMass = shapeData.initialCenterOfMass[shapeIndex];
	float3x3 Apq = make_float3x3(0.f);
	for (uint i = startParticleIndex; i <= endParticleIndex; i++)
	{
		uint particleIndex = shapeData.particleIndices[i];
		float3 p = particleData.predictedPosition[particleIndex] - centerOfMass;
		float3 q = particleData.initialPosition[particleIndex] - initialCenterOfMass;
		float mass = particleData.mass[particleIndex];

		Apq += mass * outerProduct(p, q);
	}

	// A berechnen
	float3x3 A = Apq * shapeData.Aqq[shapeIndex];
	if (parameterData.useVolumeConservation_softLinear)
	{
		float detA = det(A);
		if (detA > 0.f)
			A = (1.f / powf(detA, 1.f / 3.f)) * A;
	}

	// R berechnen
	float3x3 R = shapeData.currentRotation[shapeIndex];
	float epsilon = 0.00000001f;
	for (uint i = 0; i < parameterData.rotExtractionIterations; i++)
	{
		float3 rxa = crossProduct(R.column0, A.column0) + crossProduct(R.column1, A.column1) + crossProduct(R.column2, A.column2);
		float ra = dot(R.column0, A.column0) + dot(R.column1, A.column1) + dot(R.column2, A.column2);
		float3 w = rxa / (fabsf(ra) + epsilon);
		float angle = length(w);
		if (angle < epsilon)
			break;
		float3 axis = w / angle;
		R = make_float3x3(angle, axis) * R;
	}
	shapeData.currentRotation[shapeIndex] = R;

	uint shapeType = shapeData.type[shapeIndex];
	if (shapeType == 0) // rigid body
		shapeData.finalDeformation[shapeIndex] = R;
	else if (shapeType == 1) // soft body linear
	{
		float deformation = frobeniusNorm(A);
		if (deformation > parameterData.deformationThreshold)
		{
			float gamma = parameterData.deformationThreshold / deformation;
			A = gamma * A + (1.f - gamma) * R;
		}

		float beta = parameterData.beta_softLinear;
		shapeData.finalDeformation[shapeIndex] = beta * A + (1.f - beta) * R;
	}
	else if (shapeType == 2) // soft body quadratic
	{
		float3x9 RTilde = make_float3x9(R);

		// ApqTilde ausrechnen
		float3x9 ApqTilde = make_float3x9(0.f);
		for (uint i = startParticleIndex; i <= endParticleIndex; i++)
		{
			uint particleIndex = shapeData.particleIndices[i];
			float3 p = particleData.predictedPosition[particleIndex] - centerOfMass;
			float3 q = particleData.initialPosition[particleIndex] - initialCenterOfMass;
			float9 qTilde = make_float9(q.x, q.y, q.z, q.x * q.x, q.y * q.y, q.z * q.z, q.x * q.y, q.y * q.z, q.z * q.x);
			float mass = particleData.mass[particleIndex];

			ApqTilde += mass * outerProduct(p, qTilde);
		}

		// ATilde berechnen
		float3x9 ATilde = ApqTilde * shapeData.AqqTilde[shapeIndex];

		float deformation = frobeniusNorm(ATilde);
		if (deformation > 3.f * parameterData.deformationThreshold)
		{
			float gamma = (3.f * parameterData.deformationThreshold) / deformation;
			ATilde = gamma * ATilde + (1.f - gamma) * RTilde;
		}

		float beta = parameterData.beta_softQuadratic;
		shapeData.finalDeformationTilde[shapeIndex] = beta * ATilde + (1.f - beta) * RTilde;
	}
}