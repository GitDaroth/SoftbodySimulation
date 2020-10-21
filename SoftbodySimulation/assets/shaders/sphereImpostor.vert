#version 330

layout(location = 0) in vec3 position;

uniform mat4 viewMatrix;
uniform vec3 color;

out VertexShaderData
{
	vec3 spherePosition;
	float radius;
	vec4 color;
	vec3 lightDirection;
} outData;

void main()
{
	outData.spherePosition = (viewMatrix * vec4(position, 1.f)).xyz;
	outData.radius = 0.075f;
	outData.color = vec4(color, 1.0f);
	outData.lightDirection = (viewMatrix * vec4(1.f, 1.f, 1.f, 0.f)).xyz;
}