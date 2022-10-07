#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <GL/glut.h>

namespace py = pybind11;

class Object
{
private:
    float x, y, z;

public:
    Object(float x, float y, float z) : x(x), y(y), z(z) {}
};

class Env
{
private:
    py::array_t<uint8_t> rgb_buf;
    py::array_t<float_t> depth_buf;
    std::vector<Object> objects;

public:
    Env() : rgb_buf({90, 160, 3}), depth_buf({90, 160})
    {
        int argc = 0;
        glutInit(&argc, nullptr);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(160, 90);
        glutCreateWindow("Project 3");

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
    };

    void set_obstacles(py::array_t<float_t> obstacles)
    {
        auto r = obstacles.unchecked<2>();
        for (py::ssize_t i = 0; i < r.shape(0); i++)
        {
            objects.emplace_back(r(i, 0), r(i, 1), r(i, 2));
        }
    };

    std::pair<py::array_t<uint8_t>, py::array_t<float_t>> render()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glViewport(0, 0, 160, 90);
        gluPerspective(60.0f, 16. / 9, 0.1f, 10.0f);
        gluLookAt(0.5f, -0.5f, 0.7f,
                  0.5f, 0.5f, 0.0f,
                  0.0f, 1.0f, 0.0f);

        glFlush();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, 160, 90, GL_BGR, GL_UNSIGNED_BYTE, rgb_buf.request().ptr);
        glReadPixels(0, 0, 160, 90, GL_DEPTH_COMPONENT, GL_FLAT, depth_buf.request().ptr);
        return {rgb_buf, depth_buf};
    };
    ~Env(){};
};
