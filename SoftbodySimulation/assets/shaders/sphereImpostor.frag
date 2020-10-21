#version 330

uniform mat4 projectionMatrix;

in FragmentShaderData
{
	smooth vec3 fragPosition;
	flat vec3 spherePosition;
	flat float radius;
	flat vec4 color;
	flat vec3 lightDirection;
};

out vec4 fragColor;

void main()
{
	// depth calculation and ray testing
	vec3 rayDirection = normalize(fragPosition);
	
	float B = -2.f * dot(rayDirection, spherePosition);
	float C = dot(spherePosition, spherePosition) - (radius * radius);

	float det = (B * B) - 4.f * C;
	if(det < 0.f)
		discard;

	float sqrtDet = sqrt(det);
	float t1 = (-B + sqrtDet) * 0.5f;
	float t2 = (-B - sqrtDet) * 0.5f;

	vec3 surfacePosition = rayDirection * min(t1, t2);
	vec4 perspectiveSurfacePosition = projectionMatrix * vec4(surfacePosition, 1.f);
	float depth = perspectiveSurfacePosition.z / perspectiveSurfacePosition.w;
	gl_FragDepth = ((gl_DepthRange.diff * depth) + gl_DepthRange.near + gl_DepthRange.far) * 0.5f;
	
	// phong lighting
	vec3 normal = normalize(surfacePosition - spherePosition);

	vec3 diffuseColor = color.rgb;
	vec3 ambientColor = 0.5f * diffuseColor;
	vec3 specularColor = vec3(1.f, 1.f, 1.f);

	float ka = 1.f;
	float kd = 1.f;
	float ks = 0.35f;
	float shininess = 5.f;

	float diffuseFactor = max(dot(normal, normalize(lightDirection)), 0.f);
	float specularFactor = 0.f;

	if(diffuseFactor > 0.f)
	{
		vec3 eyeDirection = -rayDirection;
		vec3 reflectionDirection = reflect(-normalize(lightDirection), normal);
		specularFactor = pow(max(dot(reflectionDirection, eyeDirection), 0.f), shininess);
	}

	fragColor = vec4(ka * ambientColor + 
					 kd * diffuseColor * diffuseFactor + 
					 ks * specularColor * specularFactor, 1.f);
}