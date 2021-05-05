//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bertok Attila
// Neptun : I7XH6P
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f; 
	T d; 
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

const int tessellationLevel = 100;

class Quaternion {
	float real;
	float i;
	float j;
	float k;
public:
	Quaternion(float i, float j, float k, float real) {
		this->real = real;
		this->i = i;
		this->j = j;
		this->k = k;
	}
	Quaternion() {}
	Quaternion(vec4 q) {
		this->real = q.w;
		this->i = q.x;
		this->j = q.y;
		this->k = q.z;
	}
	vec4 getVec4() {
		return vec4(i, j, k, real);
	}

	static Quaternion quaternionMult(Quaternion k1, Quaternion k2) {
		Quaternion resultQuaternion;
		resultQuaternion.real = k1.real * k2.real - k1.i * k2.i - k1.j * k2.j - k1.k * k2.k;
		resultQuaternion.i = k1.i * k2.real + k2.i * k1.real + k1.j * k2.k - k1.k * k2.j;
		resultQuaternion.j = k1.j * k2.real + k2.j * k1.real + k1.k * k2.i - k1.i * k2.k;
		resultQuaternion.k = k1.k * k2.real + k2.k * k1.real + k1.i * k2.j - k1.j * k2.i;
		return resultQuaternion;
	}
};

struct Camera { 
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 90.0f * (float)M_PI / 180.0f;
		fp = 0.01; bp = 10;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct SceneCamera : Camera {
public:
	SceneCamera() {
		asp = (float)windowWidth / windowHeight;
		fp = 0.5; bp = 100;
	}
	mat4 V() {
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

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 rotateAround;
	vec3 startingPos;
	vec3 La, Le;
	vec4 wLightPos;
	void animate(float t) {
		Quaternion q = Quaternion(vec4(cosf(t / 4.0f), sinf(t / 4.0f) * cosf(t) / 2.0f, sinf(t / 4.0f) * sinf(t) / 2.0f, sinf(t / 4.0f) * sqrtf(3.0f / 4.0f)));
		Quaternion qInv = Quaternion(vec4(-cosf(t / 4.0f), -sinf(t / 4.0f) * cosf(t) / 2.0f, -sinf(t / 4.0f) * sinf(t) / 2.0f, sinf(t / 4.0f) * sqrtf(3.0f / 4.0f)));
		vec4 p = vec4(startingPos.x - rotateAround.x, startingPos.y - rotateAround.y, startingPos.z - rotateAround.z, 0);
		vec4 result = Quaternion::quaternionMult(Quaternion::quaternionMult(q, p), qInv).getVec4();
		wLightPos = result + vec4(rotateAround.x, rotateAround.y, rotateAround.z, 1);
	}
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};

class Shader : public GPUProgram {
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

class PhongShader : public Shader {
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
		uniform int isGravitySheet;

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
			if(isGravitySheet == 0){
				
				if (zCoord < -0.6) {
					if (zCoord < -1) {
						if (zCoord < -1.5) {
							if (zCoord < -2.5) {
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
		Use();
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

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw(Shader* shader ) = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
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
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw(Shader* shader) {
		shader->setUniform(true, "isGravitySheet");
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
		shader->setUniform(false, "isGravitySheet");
	}
};

class Sphere : public ParamSurface {
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
public:
	GravitySheet() { create(); }
	std::vector<Mass> masses;
	
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U * 2 - 1;
		Y = V * 2 - 1;
		Z = 0;
		
		for (int i = 0; i < masses.size(); i++)
		{		
			Z = Z + Pow(Pow(Pow(X - masses.at(i).position.x, 2) + Pow(Y - masses.at(i).position.y, 2), 0.5) + 0.01, -1) * masses.at(i).weight * -1;	
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
	void Draw(Shader* shader) {
		
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};


struct Object {
	Shader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	int id;
public:
	Object(Shader* _shader, Material* _material, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
		id = -1;
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
		shader->Bind(state);
		geometry->Draw(shader);
	}
	virtual void Animate(float tstart, float tend) {}
	virtual bool shouldBeRemoved() { return false; }
	virtual bool hasCameraAttached() { return false; }
};


struct GravitySheetObject : public Object {
	std::vector<Mass> masses;
	GravitySheetObject(Shader* _shader, Material* _material, Geometry* _geometry) : Object(_shader, _material, _geometry) {}
	void Animate(float tstart, float tend) {}
	void addMass(Mass mass) {
		masses.push_back(mass);
		((GravitySheet*)geometry)->masses.push_back(mass);
		((GravitySheet*)geometry)->create();
	}
	bool shouldBeRemoved() { return false; }
};
vec3 gravity = vec3(0, 0, -10);
struct SphereObject : public Object{
	vec3 position = vec3(-1, -1, 0);
	vec3 centerPosition = vec3(0, 0, 0);
	vec3 velocity = vec3(0, 0, 0);
	float radius = 0;
	Camera* attachedCamera = NULL;
	GravitySheetObject* gravitySheetObject;
	bool active = false;
	
	float energy;
	SphereObject(Shader* _shader, Material* _material, Geometry* _geometry, vec3 velocity, vec3 scale, GravitySheetObject* gravityObj) : Object(_shader, _material, _geometry) {
		this->velocity = velocity;
		this->scale = scale;
		radius = 1.0f * scale.x;
		centerPosition = vec3(-1.0 + radius, -1.0 + radius, radius);
		gravitySheetObject = gravityObj;
		position = vec3(-1.0 + radius, -1.0 + radius, ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(-1.0 + radius, -1.0 + radius)));
		
	}
	void Animate(float tstart, float tend) { 
		vec3 positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		if (active) {
			float dt = (tend - tstart);
			vec3 force = gravity - dot(gravity, positionNormal) * positionNormal;
			vec3 lookAt = position;
			velocity = velocity + force * dt;
			position = position + velocity * dt;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y));

			lookAt = normalize(position - lookAt);
			velocity = normalize(velocity);
			velocity = velocity * sqrtf(2.0f * (energy - length(gravity) * position.z));
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));

			if (attachedCamera != NULL) {
				vec3 normalizedVelocity = normalize(velocity);
				attachedCamera->wEye = centerPosition;
				attachedCamera->wLookat = centerPosition + normalizedVelocity;
				attachedCamera->wVup = positionNormal;
			}
			
		}
		if (position.x > 1 + radius) {
			position.x = -1 - radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y));
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.x < -1 - radius) {
			position.x = 1 + radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y));
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.y > 1 + radius) {
			position.y = -1 - radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y));
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		if (position.y < -1 - radius) {
			position.y = 1 + radius;
			position.z = ((GravitySheet*)gravitySheetObject->geometry)->getZ(vec2(position.x, position.y));
			positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		}
		centerPosition = position + radius * positionNormal;
		translation = centerPosition;
		
	}
	bool hasCameraAttached() {
		return attachedCamera != NULL;
	}
	bool shouldBeRemoved() { return position.z < -1.7; }
	void attachCamera(Camera* camera) {
		attachedCamera = camera;
		attachedCamera->wEye = centerPosition;
		attachedCamera->wLookat = vec3(1,1,0);
		vec3 positionNormal = ((GravitySheet*)gravitySheetObject->geometry)->getNormal(vec2(position.x, position.y));
		attachedCamera->wVup = positionNormal;
		scale = 0;
	}
	void removeCamera() {
		attachedCamera = NULL;
		scale = vec3(0.04f, 0.04f, 0.04f);
	}
};

class Scene {
	SceneCamera camera;
	Camera followerCamera;
	bool followingSpere = false;
	std::vector<Light> lights;
public:
	std::vector<Object*> objects;
	GravitySheetObject* gravitySheetObject;
	void Build() {

		Shader* phongShader = new PhongShader();

		Material* material0 = new Material;
		material0->kd = vec3(0.26f, 0.53f, 0.96f) * 0.6;
		material0->ks = vec3(0, 0, 0);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 10;

		Geometry* sphere = new Sphere();
		Geometry* gravitySheet = new GravitySheet();
		
		gravitySheetObject = new GravitySheetObject(phongShader, material0, gravitySheet);
		gravitySheetObject->translation = vec3(0, 0, 0);
		gravitySheetObject->scale = vec3(1.0f, 1.0f, 1.0f);
		objects.push_back(gravitySheetObject);

		camera.wEye = vec3(0, 0, 40);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		followerCamera.wVup = vec3(0, 0, 1);

		lights.resize(2);
		lights[0].wLightPos = vec4(-4, 0, 1, 0.5);
		lights[0].startingPos = vec3(-4, 0, 0.5);
		lights[0].rotateAround = vec3(1, 2, 0.5);
		lights[0].La = vec3(0.1f, 0.1f, 0.1f);
		lights[0].Le = vec3(2.6, 2.6, 2.6);
		
		lights[1].wLightPos = vec4(1, 2, 0.5, 1);
		lights[1].startingPos = vec3(1, 2, 0.5);
		lights[1].rotateAround = vec3(-4, 0, 0.5);
		lights[1].La = vec3(0.1f, 0.1f, 0.1f);
		lights[1].Le = vec3(1.0, 1.0, 1.0);
		
		addNewSphere();
	}

	void Render() {
		RenderState state;
		if (followingSpere) {
			state.wEye = followerCamera.wEye;
			state.V = followerCamera.V();
			state.P = followerCamera.P();
		}
		else {
			state.wEye = camera.wEye;
			state.V = camera.V();
			state.P = camera.P();
		}
		
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}
	void switchCamera(bool space) {
		if (space) {
			if (!followingSpere) {

				((SphereObject*)objects.at(1))->attachCamera(&followerCamera);
				sphereObjectCameraOwner = ((SphereObject*)objects.at(1));
				followingSpere = true;
			}
			else {
				sphereObjectCameraOwner->removeCamera();
				followingSpere = false;
			}
		}
		else {
			((SphereObject*)objects.at(1))->attachCamera(&followerCamera);
			sphereObjectCameraOwner = ((SphereObject*)objects.at(1));
			followingSpere = true;
		}
	}

	void Animate(float tstart, float tend) {
		for (int i = 0; i < objects.size(); i++)
		{
			objects.at(i)->Animate(tstart, tend);
			if (objects.at(i)->shouldBeRemoved()) {
				bool switchCameraB = false;
				if (objects.at(i)->hasCameraAttached()) switchCameraB = true;
				objects.erase(objects.begin() + i);
				if (switchCameraB) switchCamera(false);
			}
			
		}
		for (int i = 0; i < lights.size(); i++)
		{
			lights.at(i).animate(tend);
		}

	}

	SphereObject* sphereObjectToStart = NULL;
	SphereObject* sphereObjectCameraOwner = NULL;
	void startNewSphere(vec3 velocity) {
		sphereObjectToStart->velocity = velocity;
		sphereObjectToStart->active = true;
		sphereObjectToStart->energy = length(gravity) * sphereObjectToStart->position.z + 0.5 * pow(length(velocity), 2);
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

		SphereObject* sphereObject = new SphereObject(phongShader, material, sphere, vec3(0,0,0), vec3(0.04f, 0.04f, 0.04f), gravitySheetObject);
		sphereObject->translation = vec3(0, 0, 0);
		
		sphereObjectToStart = sphereObject;
		objects.push_back(sphereObject);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) { 
	if (key == ' ') { scene.switchCamera(true);}
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

float weightCounter = 0.02;
void onMouse(int button, int state, int pX, int pY) {
	if (state) return;
	pY = -1 * (pY - windowHeight);
	float normalizedX = (float)pX / (float)windowWidth;
	float normalizedY = (float)pY / (float)windowHeight;
	if (!button) {
		scene.startNewSphere(vec3(normalizedX, normalizedY, 0));
		vec3 tmp = ((GravitySheet*)scene.gravitySheetObject->geometry)->getNormal(vec2(normalizedX * 2 - 1, normalizedY * 2 - 1));
	}
	else {
		weightCounter += 0.02;
		scene.gravitySheetObject->addMass(Mass(weightCounter, vec2(normalizedX *2 -1, normalizedY * 2 -1)));
	}
}

void onMouseMotion(int pX, int pY) {}

void onIdle() {
	static float tend = 0;
	const float dt = 0.01f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}