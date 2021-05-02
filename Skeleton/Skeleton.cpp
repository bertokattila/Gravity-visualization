//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 200;


struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.1; bp = 20;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct SceneCamera : Camera { // 3D camera
public:
	SceneCamera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 50;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -2 / (bp - fp), 0,
			0, 0, 0, 1);
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
	void animate(float dt) {
		wLightPos.x = wLightPos.x * cos(dt) - wLightPos.y * sin(dt);
		wLightPos.y = wLightPos.x * sin(dt) + wLightPos.y * cos(dt);

	}
};

//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class GouraudShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out float zCoord;           // world space z coord
		

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			zCoord = wPos.z;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in float zCoord;
		
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
		
			float discreteDarkness = 1;
			if (zCoord < -0.5) {
				if (zCoord < -0.6) {
					if (zCoord < -0.8) {
						if (zCoord < -1) {
							discreteDarkness = 0;
						}
						else {
							discreteDarkness = 0.2;
						}
					}
					else {
						discreteDarkness = 0.5;
					}
				}
				else
				{
					discreteDarkness = 0.7;
				}
			}
			

			vec3 ka = material.ka;
			vec3 kd = material.kd * discreteDarkness;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniform(state.lights[0].wLightPos, "wLightPos");
	}
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw(Shader* shader, RenderState state ) = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};
protected:
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw(Shader* shader, RenderState state) {
		shader->Bind(state);

		float zCoord;
		float discreteDarkness;

			

		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

struct Mass {
	float weight;
	vec2 position;

	Mass(float w, vec2 pos) {
		this->weight = w;
		this->position = pos;
	}
};


class GravitySheet : public ParamSurface {
	//---------------------------
public:
	GravitySheet() { create(); }
	std::vector<Mass> masses;
	
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U * 2 - 1;
		Y = V * 2 - 1;
		Z = 0;
		
		for (int i = 0; i < masses.size(); i++)
		{		
			Z = Z + Pow(Pow(Pow(X - masses.at(i).position.x, 2) + Pow(Y - masses.at(i).position.y, 2), 0.5) + 0.02, -1) * masses.at(i).weight * -1;	
		}
	}

	vec3 getNormal(vec2 from) {
		Dnum2 X(from.x, vec2(1, 0));
		Dnum2 Y(from.y, vec2(0, 1));
		Dnum2 Z = 0;
		for (int i = 0; i < masses.size(); i++)
		{
			Z = Z + Pow(Pow(Pow(X - masses.at(i).position.x, 2) + Pow(Y - masses.at(i).position.y, 2), 0.5) + 0.02, -1) * masses.at(i).weight * -1;
		}
		return normalize(vec3(-Z.d.x, -Z.d.y, 1));
	}
	float getZ(vec2 from) {
		Dnum2 X(from.x, vec2(1, 0));
		Dnum2 Y(from.y, vec2(0, 1));
		Dnum2 Z = 0;
		for (int i = 0; i < masses.size(); i++)
		{
			Z = Z + Pow(Pow(Pow(X - masses.at(i).position.x, 2) + Pow(Y - masses.at(i).position.y, 2), 0.5) + 0.02, -1) * masses.at(i).weight * -1;
		}
		return Z.f;
	}
	void Draw(Shader* shader, RenderState state) {
		shader->Bind(state);
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Tractricoid : public ParamSurface {
	//---------------------------
public:
	Tractricoid() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float height = 3.0f;
		U = U * height, V = V * 2 * M_PI;
		X = Cos(V) / Cosh(U); Y = Sin(V) / Cosh(U); Z = U - Tanh(U);
	}
};

//---------------------------
class Cylinder : public ParamSurface {
	//---------------------------
public:
	Cylinder() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = V * 2 - 1.0f;
		X = Cos(U); Y = Sin(U); Z = V;
	}
};

//---------------------------
class Torus : public ParamSurface {
	//---------------------------
public:
	Torus() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float R = 1, r = 0.5f;
		U = U * 2.0f * M_PI, V = V * 2.0f * M_PI;
		Dnum2 D = Cos(U) * r + R;
		X = D * Cos(V); Y = D * Sin(V); Z = Sin(U) * r;
	}
};

//---------------------------
class Mobius : public ParamSurface {
	//---------------------------
public:
	Mobius() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float R = 1, width = 0.5f;
		U = U * M_PI, V = (V - 0.5f) * width;
		X = (Cos(U) * V + R) * Cos(U * 2);
		Y = (Cos(U) * V + R) * Sin(U * 2);
		Z = Sin(U) * V;
	}
};

//---------------------------
class Klein : public ParamSurface {
	//---------------------------
	const float size = 1.5f;
public:
	Klein() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * M_PI * 2, V = V * M_PI * 2;
		Dnum2 a = Cos(U) * (Sin(U) + 1) * 0.3f;
		Dnum2 b = Sin(U) * 0.8f;
		Dnum2 c = (Cos(U) * (-0.1f) + 0.2f);
		X = a + c * ((U.f > M_PI) ? Cos(V + M_PI) : Cos(U) * Cos(V));
		Y = b + ((U.f > M_PI) ? 0 : c * Sin(U) * Cos(V));
		Z = c * Sin(V);
	}
};

//---------------------------
class Boy : public ParamSurface {
	//---------------------------
public:
	Boy() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = (U - 0.5f) * M_PI, V = V * M_PI;
		float r2 = sqrt(2.0f);
		Dnum2 denom = (Sin(U * 3) * Sin(V * 2) * (-3 / r2) + 3) * 1.2f;
		Dnum2 CosV2 = Cos(V) * Cos(V);
		X = (Cos(U * 2) * CosV2 * r2 + Cos(U) * Sin(V * 2)) / denom;
		Y = (Sin(U * 2) * CosV2 * r2 - Sin(U) * Sin(V * 2)) / denom;
		Z = (CosV2 * 3) / denom;
	}
};

//---------------------------
class Dini : public ParamSurface {
	//---------------------------
	Dnum2 a = 1.0f, b = 0.15f;
public:
	Dini() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 4 * M_PI, V = V * (1 - 0.1f) + 0.1f;
		X = a * Cos(U) * Sin(V);
		Y = a * Sin(U) * Sin(V);
		Z = a * (Cos(V) + Log(Tan(V / 2))) + b * U + 3;
	}
};

//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader* _shader, Material* _material, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		
		
		geometry->Draw(shader, state);
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
	virtual bool shouldBeRemoved() { return false; }
};


struct GravitySheetObject : public Object {
	std::vector<Mass> masses;
	GravitySheetObject(Shader* _shader, Material* _material, Geometry* _geometry) : Object(_shader, _material, _geometry) {

	}
	void Animate(float tstart, float tend) {

	}
	void addMass(Mass mass) {
		masses.push_back(mass);
		((GravitySheet*)geometry)->masses.push_back(mass);
		((GravitySheet*)geometry)->create();
		//Draw();

	}
	bool shouldBeRemoved() { return false; }
};

struct SphereObject : public Object{
	vec3 position = vec3(-1, -1, 0);
	vec3 centerPosition = vec3(0, 0, 0);
	vec3 velocity = vec3(0, 0, 0);
	float radius = 0;
	Camera* attachedCamera = NULL;
	GravitySheetObject* gravitySheetObject;
	bool active = false;
	vec3 gravity = vec3(0, 0, -1);
	SphereObject(Shader* _shader, Material* _material, Geometry* _geometry, vec3 velocity, vec3 scale, GravitySheetObject* gravityObj) : Object(_shader, _material, _geometry) {
		this->velocity = velocity;
		this->scale = scale;
		radius = 1.0f * scale.x;
		centerPosition = vec3(-1.0 + radius, -1.0 + radius, radius);
		position = vec3(-1.0 + radius, -1.0 + radius, 0);
		gravitySheetObject = gravityObj;
	}
	void Animate(float tstart, float tend) { 
		vec3 positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		if (active) {
			float dt = (tend - tstart);

			vec3 force = gravity - dot(gravity, positionNormal) * positionNormal;

			velocity = velocity + force * dt;

			position = position + velocity * dt;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y)); /// korrekcio

			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.x > 1 + radius) {
			position.x = -1 - radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y)); /// korrekcio
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.x < -1 - radius) {
			position.x = 1 + radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y)); /// korrekcio
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.y > 1 + radius) {
			position.y = -1 - radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y)); /// korrekcio
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.y < -1 - radius) {
			position.y = 1 + radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y)); /// korrekcio
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}

		if (attachedCamera != NULL && active) {
			vec3 normalizedVelocity = normalize(velocity);
			attachedCamera->wEye = centerPosition + normalizedVelocity * 0.01 + vec3(0, 0, 0.04);
			attachedCamera->wLookat = centerPosition + velocity;
			attachedCamera->wVup = positionNormal;
			//printf("x %f y %f z %f\n", attachedCamera->wEye.x, attachedCamera->wEye.y, attachedCamera->wEye.z);
			//printf("x %f y %f z %f\n", attachedCamera->wLookat.x, attachedCamera->wLookat.y, attachedCamera->wLookat.z);
		}

		centerPosition = position + radius * positionNormal;
		translation = centerPosition;
		
	}
	bool shouldBeRemoved() {
		//printf("pos %f\n", position.z);
		return position.z < -2; }
	void attachCamera(Camera* camera) {
		attachedCamera = camera;
		attachedCamera->wEye = centerPosition + vec3(0.01, 0.01, 0.04);
		attachedCamera->wLookat = vec3(1,1,0);
	}
	void removeCamera() {
		attachedCamera = NULL;
	}
};


//---------------------------
class Scene {
	//---------------------------
	
	SceneCamera camera;
	Camera folowerCamera;
	bool folowingSpere = false;
	std::vector<Light> lights;
public:
	std::vector<Object*> objects;
	GravitySheetObject* gravitySheetObject;
	void Build() {

		// Shaders
		Shader* phongShader = new PhongShader();
		Shader* gouraudShader = new GouraudShader();
		Shader* nprShader = new NPRShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.26f, 0.53f, 0.96f);
		material0->ks = vec3(0, 0, 0);
		material0->ka = vec3(0.5f, 0.5f, 0.5f);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 100;

	
		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* gravitySheet = new GravitySheet();
		Geometry* tractricoid = new Tractricoid();
		Geometry* torus = new Torus();
		Geometry* mobius = new Mobius();
		Geometry* klein = new Klein();
		Geometry* boy = new Boy();
		Geometry* dini = new Dini();

		// Create objects by setting up their vertex data on the GPU
		//Object* sphereObject1 = new SphereObject(phongShader, material0, sphere, vec3(0.2, 0.2, 0.2), vec3(0.05f, 0.05f, 0.05f));
		//sphereObject1->translation = vec3(0, 0, 0);
		//objects.push_back(sphereObject1);

		gravitySheetObject = new GravitySheetObject(phongShader, material0, gravitySheet);
		gravitySheetObject->translation = vec3(0, 0, 0);
		gravitySheetObject->scale = vec3(1.0f, 1.0f, 1.0f);
		objects.push_back(gravitySheetObject);

		// Create objects by setting up their vertex data on the GPU
		Object* tractiObject1 = new Object(phongShader, material0, tractricoid);
		tractiObject1->translation = vec3(-6, 3, 0);
		tractiObject1->rotationAxis = vec3(1, 0, 0);
		//objects.push_back(tractiObject1);

		Object* torusObject1 = new Object(phongShader, material0, torus);
		torusObject1->translation = vec3(-3, 3, 0);
		torusObject1->scale = vec3(0.7f, 0.7f, 0.7f);
		torusObject1->rotationAxis = vec3(1, 0, 0);
		//objects.push_back(torusObject1);

		Object* mobiusObject1 = new Object(phongShader, material0, mobius);
		mobiusObject1->translation = vec3(0, 3, 0);
		mobiusObject1->scale = vec3(0.7f, 0.7f, 0.7f);
		mobiusObject1->rotationAxis = vec3(1, 0, 0);
		//objects.push_back(mobiusObject1);

		Object* kleinObject1 = new Object(phongShader, material1, klein);
		kleinObject1->translation = vec3(3, 3, 0);
		//objects.push_back(kleinObject1);

		Object* boyObject1 = new Object(phongShader, material1, boy);
		boyObject1->translation = vec3(6, 3, 0);
		//objects.push_back(boyObject1);

		Object* diniObject1 = new Object(phongShader, material1, dini);
		diniObject1->translation = vec3(9, 3, 0);
		diniObject1->scale = vec3(0.7f, 0.7f, 0.7f);
		diniObject1->rotationAxis = vec3(1, 0, 0);
		//objects.push_back(diniObject1);

		/*int nObjects = objects.size();
		for (int i = 0; i < nObjects; i++) {
			Object* object = new Object(*objects[i]);
			object->translation.y -= 3;
			object->shader = gouraudShader;
			objects.push_back(object);
			object = new Object(*objects[i]);
			object->translation.y -= 6;
			object->shader = nprShader;
			objects.push_back(object);
		}
		*/

		// Camera
		camera.wEye = vec3(0, 0, 5);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		folowerCamera.wVup = vec3(0, 0, 1);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(1, 1, 1, 1);	// ideal point -> directional light source
		lights[0].La = vec3(0.6f, 0.6f, 0.6f);
		lights[0].Le = vec3(0.4, 0.4, 0.4);

		/*lights[1].wLightPos = vec4(5, 10, 20, 0);	// ideal point -> directional light source
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(-5, 5, 5, 0);	// ideal point -> directional light source
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 3);
		*/
		addNewSphere();
	}

	void Render() {
		RenderState state;
		if (folowingSpere) {
			state.wEye = folowerCamera.wEye;
			state.V = folowerCamera.V();
			state.P = folowerCamera.P();
		}
		else {
			state.wEye = camera.wEye;
			state.V = camera.V();
			state.P = camera.P();
		}
		
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (int i = 0; i < objects.size(); i++)
		{
			objects.at(i)->Animate(tstart, tend);
			if (objects.at(i)->shouldBeRemoved()) {
				objects.erase(objects.begin() + i);
			}
			
		}
		for (int i = 0; i < lights.size(); i++)
		{
			lights.at(i).animate(tend - tstart);

		}

	}

	SphereObject* sphereObjectToStart = NULL;
	SphereObject* sphereObjectCameraOwner = NULL;
	void startNewSphere(vec3 velocity) {
		sphereObjectToStart->velocity = velocity;
		sphereObjectToStart->active = true;
		addNewSphere();
	}
	float random() {
		return (float)rand() / (float)RAND_MAX;
	}
	void addNewSphere() {
		Shader* phongShader = new PhongShader();
		Geometry* sphere = new Sphere();
		Material* material = new Material;
		material->kd = vec3(random(), random(), random());
		material->ks = vec3(4, 4, 4);
		material->ka = vec3(0.1f, 0.1f, 0.1f);
		material->shininess = 100;

		SphereObject* sphereObject = new SphereObject(phongShader, material, sphere, vec3(0,0,0), vec3(0.05f, 0.05f, 0.05f), gravitySheetObject);
		sphereObject->translation = vec3(0, 0, 0);
		
		sphereObjectToStart = sphereObject;
		objects.push_back(sphereObject);
	}
	void switchCamera() {
		if (!folowingSpere) {
			sphereObjectToStart->attachCamera(&folowerCamera);
			sphereObjectCameraOwner = sphereObjectToStart;
		}
		else
		{
			sphereObjectCameraOwner->removeCamera();
		}
		folowingSpere = !folowingSpere;
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
	if (key == ' ') {
		scene.switchCamera();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
float weightCounter = 0.05;
void onMouse(int button, int state, int pX, int pY) {
	if (state) return;
	pY = -1 * (pY - windowHeight);
	float normalizedX = (float)pX / (float)windowWidth;
	float normalizedY = (float)pY / (float)windowHeight;
	if (!button) {
		scene.startNewSphere(vec3(normalizedX, normalizedY, 0));
		vec3 tmp = ((GravitySheet*)scene.gravitySheetObject->geometry)->getNormal(vec2(normalizedX * 2 - 1, normalizedY * 2 - 1));
		printf("x %f y %f z %f\n", tmp.x, tmp.y, tmp.z);
	}
	else {
		weightCounter += 0.005;
		scene.gravitySheetObject->addMass(Mass(weightCounter, vec2(normalizedX *2 -1, normalizedY * 2 -1)));
	}
	
	//printf("x %f y %f\n", worldX, worldY);
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}