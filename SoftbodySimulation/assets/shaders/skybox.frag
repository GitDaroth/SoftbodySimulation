#version 330

uniform samplerCube skybox;

in vec3 sampleDirection;

out vec4 fragColor;

void main()
{
	fragColor = texture(skybox, normalize(sampleDirection));
}