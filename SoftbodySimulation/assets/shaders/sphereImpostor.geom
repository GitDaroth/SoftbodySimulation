#version 330

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 projectionMatrix;

in VertexShaderData
{
	vec3 spherePosition;
	float radius;
	vec4 color;
	vec3 lightDirection;
} inData[];

out FragmentShaderData
{
	smooth vec3 fragPosition;
	flat vec3 spherePosition;
	flat float radius;
	flat vec4 color;
	flat vec3 lightDirection;
};

void main()
{
	vec4 cornerPosition;
	float quadScale = 1.5f;

	//Bottom-left
	spherePosition = inData[0].spherePosition;
	radius = inData[0].radius;
	color = inData[0].color;
	lightDirection = inData[0].lightDirection;
	cornerPosition = vec4(inData[0].spherePosition, 1.f);
	cornerPosition.xy += vec2(-inData[0].radius, -inData[0].radius) * quadScale;
	fragPosition = cornerPosition.xyz;
	gl_Position = projectionMatrix * cornerPosition;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();

	//Top-left
	spherePosition = inData[0].spherePosition;
	radius = inData[0].radius;
	color = inData[0].color;
	lightDirection = inData[0].lightDirection;
	cornerPosition = vec4(inData[0].spherePosition, 1.f);
	cornerPosition.xy += vec2(-inData[0].radius, inData[0].radius) * quadScale;
	fragPosition = cornerPosition.xyz;
	gl_Position = projectionMatrix * cornerPosition;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();

	//Bottom-right
	spherePosition = inData[0].spherePosition;
	radius = inData[0].radius;
	color = inData[0].color;
	lightDirection = inData[0].lightDirection;
	cornerPosition = vec4(inData[0].spherePosition, 1.f);
	cornerPosition.xy += vec2(inData[0].radius, -inData[0].radius) * quadScale;
	fragPosition = cornerPosition.xyz;
	gl_Position = projectionMatrix * cornerPosition;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();

	//Top-right
	spherePosition = inData[0].spherePosition;
	radius = inData[0].radius;
	color = inData[0].color;
	lightDirection = inData[0].lightDirection;
	cornerPosition = vec4(inData[0].spherePosition, 1.f);
	cornerPosition.xy += vec2(inData[0].radius, inData[0].radius) * quadScale;
	fragPosition = cornerPosition.xyz;
	gl_Position = projectionMatrix * cornerPosition;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
}