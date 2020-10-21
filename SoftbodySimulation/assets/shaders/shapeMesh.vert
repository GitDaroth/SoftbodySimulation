#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 fragNormal;
out vec3 lightDirection;
out vec3 eyeDirection;

void main()
{
	fragNormal = (viewMatrix * modelMatrix * vec4(normal, 0.f)).xyz;
	lightDirection = (viewMatrix * normalize(vec4(1.f, 0.8f, 0.3f, 0.f))).xyz;
	eyeDirection = -(viewMatrix * vec4(position, 1.0)).xyz;
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}