#version 330

uniform sampler2D diffuseTexture;
uniform sampler2D normalTexture;
uniform sampler2D specularTexture;
uniform mat4 viewMatrix;

smooth in vec2 uvFrag;
in vec3 lightDirection;
in vec3 eyeDirection;
in mat3 TBN;

out vec4 fragColor;

void main()
{
	vec3 normalTexel = texture2D(normalTexture, uvFrag).rgb;
    vec3 vertexTBNNormal = normalize(normalTexel * 2.f - 1.f);
    vec3 normal = (viewMatrix * vec4(normalize(TBN * vertexTBNNormal), 0.f)).xyz;

	vec3 diffuseColor = texture2D(diffuseTexture, uvFrag).rgb;
	vec3 ambientColor = 0.5f * diffuseColor;
	vec3 specularColor = texture2D(specularTexture, uvFrag).rgb;

	float ka = 1.f;
	float kd = 1.f;
	float ks = 1.f;
	float shininess = 128.f;

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