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
    Vector3f p;
    std::unique_ptr<Geometry> geometry;
    Mesh(Geometry *geometry, Vector3f p) : p(p), geometry(geometry) {}
    void draw()
    {
        glPushMatrix();
        glTranslated(p.x, p.y, p.z);
        geometry->draw();
        glPopMatrix();
    }
    Vector3f nearestPt(Vector3f const &camera)
    {
        return geometry->nearestPt(camera - p) + p;
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
        glutSolidSphere(r, 10, 8);
    }
    Vector3f nearestPt(Vector3f const &p)
    {
        return p / fmaxf(p.norm(), 1) * r;
    }
};
