#pragma once

#include <random>
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

typedef std::vector<Mesh> env_t;

class Env
{
private:
    py::array_t<uint8_t> rgb;
    py::array_t<float_t> depth;
    py::array_t<float_t> nearest_pt;
    std::vector<env_t> envs;
    int n_envs;

public:
    Env(int n_envs) : n_envs(n_envs), rgb({n_envs, 90, 160, 3}), depth({n_envs, 90, 160}), nearest_pt({n_envs, 3})
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

    void set_obstacles()
    {
        envs.clear();
        envs.resize(n_envs);
        for (int i = 0; i < n_envs; i++)
        {
            for (int j = 0; j < 40; j++)
            {
                float x = float(rand()) / RAND_MAX * 30 + 5;
                float y = float(rand()) / RAND_MAX * 10 - 5;
                float z = float(rand()) / RAND_MAX * 8 - 2;
                float r = float(rand()) / RAND_MAX + 0.5;
                Geometry *m;
                switch (rand() % 5)
                {
                case 0:
                case 1:
                    m = new Cube(r);
                    break;
                case 2:
                case 3:
                    m = new Ball(r);
                    break;
                case 4:
                    m = new Cone(r / 2);
                    z = -1;
                    break;

                default:
                    break;
                }
                envs[i].emplace_back(m, Vector3f{x, y, z});
            }
        }
    };

    std::tuple<py::array_t<uint8_t>, py::array_t<float_t>, py::array_t<float_t>> render(
        py::array_t<float_t> cameras, bool flush)
    {
        assert(cameras.shape(0) == n_envs);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        auto r = cameras.unchecked<2>();
        auto nearest_pt_ptr = nearest_pt.mutable_unchecked<2>();

        auto n_envs = envs.size();
        for (int i = 0; i < cameras.shape(0); i++)
        {
            glLoadIdentity();
            glViewport(0, 90 * i, 160, 90);
            gluPerspective(180 * 0.35, 16. / 9, 0.01f, 10.0f);
            set_camera(r(i, 0), r(i, 1), r(i, 2), r(i, 3), r(i, 4), r(i, 5));

            Vector3f camera_p{r(i, 0), r(i, 1), r(i, 2)};
            nearest_pt_ptr(i, 0) = camera_p.x;
            nearest_pt_ptr(i, 1) = camera_p.y;
            nearest_pt_ptr(i, 2) = fmin(-1, camera_p.z);
            float nearest_distance = fabs(camera_p.z - fmin(-1, camera_p.z));

            glBegin(GL_QUADS);
            glVertex3f(-10, -10, -1);
            glVertex3f(40, -10, -1);
            glVertex3f(40, 10, -1);
            glVertex3f(-10, 10, -1);
            glEnd();

            for (auto &ball : envs[i])
            {
                ball.draw();
                Vector3f pt = ball.nearestPt(camera_p);
                float distance = (pt - camera_p).norm();
                if (distance < nearest_distance)
                {
                    nearest_pt_ptr(i, 0) = pt.x;
                    nearest_pt_ptr(i, 1) = pt.y;
                    nearest_pt_ptr(i, 2) = pt.z;
                    nearest_distance = distance;
                }
            }
        }
        if (flush)
            glFlush();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, 160, 90 * n_envs, GL_BGR, GL_UNSIGNED_BYTE, rgb.request().ptr);
        glReadPixels(0, 0, 160, 90 * n_envs, GL_DEPTH_COMPONENT, GL_FLOAT, depth.request().ptr);
        return {rgb, depth, nearest_pt};
    };
    ~Env(){};
};
