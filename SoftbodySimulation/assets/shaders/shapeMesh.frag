#version 330

in vec3 fragNormal;
in vec3 lightDirection;
in vec3 eyeDirection;

uniform vec3 color;
uniform bool usePhongShading;

out vec4 fragColor;

void main()
{
	if(usePhongShading)
	{
		vec3 diffuseColor = color;
		vec3 ambientColor = 0.5f * diffuseColor;
		vec3 specularColor = vec3(1.f, 1.f, 1.f);

		float ka = 1.f;
		float kd = 1.f;
		float ks = 0.3f;
		float shininess = 32.f;

		vec3 normal = normalize(fragNormal);
		float diffuseFactor = max(dot(normal, normalize(lightDirection)), 0.f);
		float specularFactor = 0.f;

		if(diffuseFactor > 0.f)
		{
			vec3 reflectionDirection = reflect(normalize(-lightDirection), normal);
			specularFactor = pow(max(dot(reflectionDirection, normalize(eyeDirection)), 0.f), shininess);
		}

		fragColor = vec4(ka * ambientColor + 
						 kd * diffuseColor * diffuseFactor + 
						 ks * specularColor * specularFactor, 1.f);
	}
	else
	{
		fragColor = vec4(color, 1.f);
	}
}