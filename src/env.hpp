#pragma once

#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "objects/ext.hpp"

namespace py = pybind11;

inline void set_camera(
    double x, double y, double z, double roll, double pitch, double yaw)
{
    double cx = cos(roll);
    double cy = cos(pitch);
    double cz = cos(yaw);
    double sx = sin(roll);
    double sy = sin(pitch);
    double sz = sin(yaw);

    double up_x = cx * cz * sy + sx * sz;
    double up_y = -cz * sx + cx * sy * sz;
    double up_z = cx * cy;
    double forward_x = cy * cz;
    double forward_y = cy * sz;
    double forward_z = -sy;

    gluLookAt(x, y, z,
              x + forward_x, y + forward_y, z + forward_z,
              up_x, up_y, up_z);
}

typedef std::vector<Ball> env_t;

class Env
{
private:
    py::array_t<uint8_t> rgb_buf;
    py::array_t<float_t> depth_buf;
    std::vector<env_t> envs;
    int n_envs;

public:
    Env(int n_envs) : n_envs(n_envs), rgb_buf({n_envs, 90, 160, 3}), depth_buf({n_envs, 90, 160})
    {
        int argc = 0;
        glutInit(&argc, nullptr);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(160, 90 * n_envs);
        glutCreateWindow("quadsim");

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // glEnable(GL_BLEND);

        //材质反光性设置
        GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0}; //镜面反射参数
        GLfloat mat_shininess[] = {50.0};              //高光指数
        GLfloat light_position[] = {-1.0, -1.0, 5.0, 0.0};
        GLfloat white_light[] = {1.0, 1.0, 1.0, 1.0};         //灯位置(1,1,1), 最后1-开关
        GLfloat Light_Model_Ambient[] = {0.5, 0.5, 0.5, 1.0}; //环境光参数

        glClearColor(0.0, 0.0, 0.0, 0.0); //背景色
        // glShadeModel(GL_SMOOTH);          //多变性填充模式

        //材质属性
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

        //灯光设置
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);               //散射光属性
        glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);              //镜面反射光
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, Light_Model_Ambient); //环境光参数

        glEnable(GL_LIGHTING);   //开关:使用光
        glEnable(GL_LIGHT0);     //打开0#灯
        glEnable(GL_DEPTH_TEST); //打开深度测试
    };

    void set_obstacles(py::array_t<float_t> obstacles)
    {
        assert(obstacles.shape(0) == n_envs);
        envs.clear();
        auto r = obstacles.unchecked<3>();
        envs.resize(r.shape(0));
        for (py::ssize_t i = 0; i < r.shape(0); i++)
        {
            for (py::ssize_t j = 0; j < r.shape(1); j++)
            {
                envs[i].emplace_back(r(i, j, 0), r(i, j, 1), r(i, j, 2));
            }
        }
    };

    std::pair<py::array_t<uint8_t>, py::array_t<float_t>> render(
        py::array_t<float_t> cameras)
    {
        assert(cameras.shape(0) == n_envs);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        auto r = cameras.unchecked<2>();
        auto n_envs = envs.size();
        for (int i = 0; i < cameras.shape(0); i++)
        {
            glLoadIdentity();
            glViewport(0, 90 * i, 160, 90);
            gluPerspective(180 * 0.35, 16. / 9, 0.01f, 10.0f);
            set_camera(r(i, 0), r(i, 1), r(i, 2), r(i, 3), r(i, 4), r(i, 5));

            for (auto &ball : envs[i])
            {
                ball.draw();
            }

            glBegin(GL_QUADS);
            glVertex3f(-10, -10, -1);
            glVertex3f(40, -10, -1);
            glVertex3f(40, 10, -1);
            glVertex3f(-10, 10, -1);
            glEnd();
        }
        // glFlush();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, 160, 90 * n_envs, GL_BGR, GL_UNSIGNED_BYTE, rgb_buf.request().ptr);
        glReadPixels(0, 0, 160, 90 * n_envs, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buf.request().ptr);
        return {rgb_buf, depth_buf};
    };
    ~Env(){};
};
