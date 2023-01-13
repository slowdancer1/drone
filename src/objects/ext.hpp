#pragma once

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <memory>
#include <cmath>
#include <vector>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

struct Vector3f
{
    float x, y, z;
    float norm() const
    {
        return sqrtf(x * x + y * y + z * z);
    }
    float dot(Vector3f const &s) const
    {
        return x * s.x + y * s.y + z * s.z;
    }
    Vector3f operator+(Vector3f const &s) const { return Vector3f{x + s.x, y + s.y, z + s.z}; }
    Vector3f operator-(Vector3f const &s) const { return Vector3f{x - s.x, y - s.y, z - s.z}; }
    Vector3f operator*(float const &s) const { return Vector3f{x * s, y * s, z * s}; }
    Vector3f operator/(float const &s) const { return Vector3f{x / s, y / s, z / s}; }
};

class Geometry
{
public:
    Geometry(){};
    virtual void draw() = 0;
    virtual Vector3f nearestPt(Vector3f const &p) = 0;
    virtual ~Geometry(){};
};

class Mesh
{
public:
    Vector3f p, v;
    std::unique_ptr<Geometry> geometry;
    Mesh(Geometry *geometry, Vector3f p, Vector3f v) : p(p), v(v), geometry(geometry) {}
    void draw()
    {
        glPushMatrix();
        glTranslated(p.x, p.y, p.z);
        geometry->draw();
        glPopMatrix();
    }
    void set_p(Vector3f p_new){
        p = p_new;
    }
    void update(float ctl_dt){
        p = p + v * ctl_dt;
    }
    Vector3f nearestPt(Vector3f const &camera)
    {
        return geometry->nearestPt(camera - p) + p;
    }
    Vector3f get_p()
    {
        return p;
    }
};

class ExtShape
{
public:
    ExtShape(std::string obj_file, std::string mtl_file);
    ~ExtShape(){};
    void draw();
    void readmtl(std::string mtl_file);
    void readobj(std::string obj_file);

private:
    std::vector<std::vector<float>> vSets;
    std::vector<std::vector<int>> fSets;
    std::map<std::string, int> mtlMap;
    std::vector<std::vector<float>> mtlSets;
};

class Ball : public Geometry
{
public:
    float r;
    Ball(float r) : r(r) {};
    void draw()
    {
        glutSolidSphere(r, 100, 100);
    }
    Vector3f nearestPt(Vector3f const &p)
    {
        return p / p.norm() * fminf(p.norm() - 0.001, r);
    }
};

class Cone : public Geometry
{
public:
    float r, h;
    Cone(float r, float h) : r(r), h(h){};
    void draw()
    {
        glutSolidCone(r, h, 100, 100);
    }
    Vector3f nearestPt(Vector3f const &p)
    {
        Vector3f _p = p;
        if (_p.z < 0) _p.z = 0;
        if (_p.z > 100) _p.z = 100;
        float _r = r * (1 - _p.z / 100);
        float norm = sqrtf(_p.x * _p.x + _p.y * _p.y);
        _p.x = _p.x / norm * fminf(norm - 0.001, _r);
        _p.y = _p.y / norm * fminf(norm - 0.001, _r);
        return _p;
    }
};

class Cube : public Geometry
{
public:
    float a;
    Cube(float a) : a(a) {};
    void draw()
    {
        glutSolidCube(a);
    }
    Vector3f nearestPt(Vector3f const &p)
    {
        Vector3f _p = p;
        float r = a / 2;
        if (_p.x > r) _p.x = r;
        else if (_p.x < -r) _p.x = -r;
        else _p.x = _p.x > 0 ? _p.x - 0.001 : _p.x + 0.001;
        if (_p.y > r) _p.y = r;
        else if (_p.y < -r) _p.y = -r;
        else _p.y = _p.y > 0 ? _p.y - 0.001 : _p.y + 0.001;
        if (_p.z > r) _p.z = r;
        else if (_p.z < -r) _p.z = -r;
        else _p.z = _p.z > 0 ? _p.z - 0.001 : _p.z + 0.001;
        return _p;
    }
};
