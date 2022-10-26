#pragma once

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include "ball.h"


class Geometry
{
public:
    Geometry(){};
    virtual void draw() = 0;
    virtual ~Geometry(){};
};

class Mesh
{
public:
    float x, y, z;
    Geometry &geometry;
    Mesh(Geometry &geometry, float x, float y, float z) : x(x), y(y), z(z), geometry(geometry) {}
    void draw(){
        glPushMatrix();
        glTranslated(x, y, z);
        geometry.draw();
        glPopMatrix();
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

class Ball : Geometry
{
public:
    Ball(){};
    void draw()
    {
        glutSolidSphere(1, 10, 8);
    }
};
