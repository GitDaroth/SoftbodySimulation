#version 330

layout(location = 0) in vec3 position;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 sampleDirection;

void main()
{
	sampleDirection = position;
	gl_Position = vec4(projectionMatrix * viewMatrix * vec4(position, 1.0)).xyww;
}