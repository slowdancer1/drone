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

Vector3f set_camera(
    Vector3f p, float roll, float pitch, float yaw)
{
    float cx = cos(roll);
    float cy = cos(pitch);
    float cz = cos(yaw);
    float sx = sin(roll);
    float sy = sin(pitch);
    float sz = sin(yaw);

    float up_x = cx * cz * sy + sx * sz;
    float up_y = -cz * sx + cx * sy * sz;
    float up_z = cx * cy;
    float forward_x = cy * cz;
    float forward_y = cy * sz;
    float forward_z = -sy;

    gluLookAt(p.x, p.y, p.z,
              p.x + forward_x, p.y + forward_y, p.z + forward_z,
              up_x, up_y, up_z);
    return {forward_x, forward_y, forward_z};
}

typedef std::vector<Mesh> env_t;

class Env
{
private:
    py::array_t<uint8_t> rgb;
    py::array_t<float_t> depth;
    py::array_t<float_t> nearest_pt;
    std::vector<env_t> envs;
    int n_envs_h, n_envs_w, n_envs;
    std::random_device rd;

public:
    Env(int n_envs_h, int n_envs_w, int width, int height) : n_envs_h(n_envs_h), n_envs_w(n_envs_w), rgb({n_envs_h, height, n_envs_w, width, 3}), depth({n_envs_h, height, n_envs_w, width}), nearest_pt({n_envs_h * n_envs_w, 3}), n_envs(n_envs_h * n_envs_w)
    {
        int argc = 0;
        glutInit(&argc, nullptr);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(width * n_envs_w, height * n_envs_h);
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
                float x = float(rd()) / rd.max() * 60 + 2;
                float y = float(rd()) / rd.max() * 10 - 5;
                float z = float(rd()) / rd.max() * 8 - 2;
                float vx = 0, vy = 0, vz = 0;
                if (float(rd()) / rd.max() < 0.5) {
                    vx = float(rd()) / rd.max() * 2 - 1;
                    vy = float(rd()) / rd.max() * 2 - 1;
                    vz = float(rd()) / rd.max() * 2 - 1;
                }
                float r = float(rd()) / rd.max();
                Geometry *m;
                switch (rd() % 5)
                {
                case 0:
                case 1:
                    m = new Cube(r + 0.1);
                    break;
                case 2:
                case 3:
                    m = new Ball(r + 0.1);
                    break;
                case 4:
                    m = new Cone(r / 4 + 0.1);
                    z = -1;
                    break;

                default:
                    break;
                }
                envs[i].emplace_back(m, Vector3f{x, y, z}, Vector3f{vx, vy, vz});
            }
        }
    };

    std::tuple<py::array_t<uint8_t>, py::array_t<float_t>, py::array_t<float_t>> render(
        py::array_t<float_t> cameras, float ctl_dt, bool flush)
    {
        int height = depth.shape(1);
        int width = depth.shape(3);
        assert(cameras.shape(0) == n_envs);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        auto r = cameras.unchecked<2>();
        auto nearest_pt_ptr = nearest_pt.mutable_unchecked<2>();

        auto n_envs = envs.size();
        for (int i = 0; i < cameras.shape(0); i++)
        {
            glLoadIdentity();
            glViewport(width * (i % n_envs_w), height * (i / n_envs_w), width, height);
            gluPerspective(180 * 0.354, 12. / 9, 0.01f, 10.0f);

            Vector3f camera_p{r(i, 0), r(i, 1), r(i, 2)};
            Vector3f forward = set_camera(camera_p, r(i, 3), r(i, 4), r(i, 5));

            nearest_pt_ptr(i, 0) = camera_p.x;
            nearest_pt_ptr(i, 1) = camera_p.y;
            nearest_pt_ptr(i, 2) = fmin(-1, camera_p.z);
            float nearest_distance = fabs(camera_p.z - fmin(-1, camera_p.z));

            for (auto &ball : envs[i])
            {
                ball.update(ctl_dt);
            }
            for (auto &ball : envs[i])
            {
                Vector3f pt = ball.nearestPt(camera_p);
                Vector3f cam2pt = pt - camera_p;
                float distance = cam2pt.norm();
                if (distance < nearest_distance)
                {
                    nearest_pt_ptr(i, 0) = pt.x;
                    nearest_pt_ptr(i, 1) = pt.y;
                    nearest_pt_ptr(i, 2) = pt.z;
                    nearest_distance = distance;
                }
                float forward_distance = cam2pt.dot(forward);
                if (0 < forward_distance && forward_distance < 10 && distance < 10) {
                    ball.draw();
                }
            }

            glBegin(GL_QUADS);
            glVertex3f(-10, -10, -1);
            glVertex3f(80, -10, -1);
            glVertex3f(80, 10, -1);
            glVertex3f(-10, 10, -1);
            glEnd();

        }
        if (flush)
            glFlush();
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width * n_envs_w, height * n_envs_h, GL_BGR, GL_UNSIGNED_BYTE, rgb.request().ptr);
        glReadPixels(0, 0, width * n_envs_w, height * n_envs_h, GL_DEPTH_COMPONENT, GL_FLOAT, depth.request().ptr);
        return {rgb, depth, nearest_pt};
    };
    ~Env(){};
};
