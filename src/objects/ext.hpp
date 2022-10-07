#ifndef EXTSHAPE_H
#define EXTSHAPE_H

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <GL/glut.h>
#include "ball.h"

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

class Ball
{
private:
    float x, y, z;
    ExtShape shape;

public:
    Ball(float x, float y, float z) : x(x), y(y), z(z), shape(BALL_3D_OBJ, BALL_3D_MTL) {}
    void draw()
    {
        glPushMatrix();
        glTranslated(x, y, z);
        // glColor3f(0.5, 0.5, 0.5);
        // shape.draw();
        glutSolidSphere(1, 10, 8);
        glPopMatrix();
    }
};

#endif
