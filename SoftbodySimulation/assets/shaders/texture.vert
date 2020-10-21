#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec3 bitangent;
layout(location = 4) in vec2 uv;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

smooth out vec2 uvFrag;
out vec3 lightDirection;
out vec3 eyeDirection;
out mat3 TBN;

void main()
{
    TBN = mat3(tangent, bitangent, normal);

	uvFrag = uv;
	lightDirection = (viewMatrix * vec4(-1.f, 1.f, -1.f, 0.f)).xyz;
	eyeDirection = -(viewMatrix * vec4(position, 1.0)).xyz;
	gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
}