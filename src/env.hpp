#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <GL/glut.h>

#include "objects/ext.hpp"

namespace py = pybind11;

inline void set_camera(
    double x, double y, double z, double r, double i, double j, double k)
{
    double two_s = 2.0 / (r * r + i * i + j * j + k * k);

        double up_x = two_s * (i * k + j * r);
        double up_y = two_s * (j * k - i * r);
        double up_z = 1 - two_s * (i * i + j * j);
        double forward_x = 1 - two_s * (j * j + k * k);
        double forward_y = two_s * (i * j + k * r);
        double forward_z = two_s * (i * k - j * r);

        gluLookAt(x, y, z,
                  x + forward_x, y + forward_y, z + forward_z,
                  up_x, up_y, up_z);
}

class Env
{
private:
    py::array_t<uint8_t> rgb_buf;
    py::array_t<float_t> depth_buf;
    std::vector<Ball> balls;

public:
    Env() : rgb_buf({90, 160, 3}), depth_buf({90, 160})
    {
        int argc = 0;
        glutInit(&argc, nullptr);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(160, 90);
        glutCreateWindow("Project 3");
        glEnable(GL_DEPTH_TEST);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
    };

    void set_obstacles(py::array_t<float_t> obstacles)
    {
        balls.clear();
        auto r = obstacles.unchecked<2>();
        for (py::ssize_t i = 0; i < r.shape(0); i++)
        {
            balls.emplace_back(r(i, 0), r(i, 1), r(i, 2));
        }
    };

    std::pair<py::array_t<uint8_t>, py::array_t<float_t>> render(
        double x, double y, double z, double r, double i, double j, double k)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glViewport(0, 0, 160, 90);
        gluPerspective(180*0.35, 16. / 9, 0.01f, 10.0f);
        set_camera(x, y, z, r, i, j, k);

        for (auto &ball : balls)
        {
            ball.draw();
        }

        glFlush();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, 160, 90, GL_BGR, GL_UNSIGNED_BYTE, rgb_buf.request().ptr);
        glReadPixels(0, 0, 160, 90, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buf.request().ptr);
        return {rgb_buf, depth_buf};
    };
    ~Env(){};
};
