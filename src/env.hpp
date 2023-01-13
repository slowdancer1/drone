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

void set_camera(
    Vector3f p, Vector3f f, Vector3f u)
{
    gluLookAt(p.x, p.y, p.z,
              p.x + f.x, p.y + f.y, p.z + f.z,
              u.x, u.y, u.z);
}

typedef std::vector<Mesh> env_t;

class Env
{
private:
    py::array_t<uint8_t> rgb;
    py::array_t<float_t> depth;
    py::array_t<float_t> nearest_pt;
    py::array_t<float_t> obstacle_pt;
    std::vector<env_t> envs;
    int n_envs_h, n_envs_w, n_envs;
    std::random_device rd;
    bool test;

public:
    Env(int n_envs_h, int n_envs_w, int width, int height, bool test) : n_envs_h(n_envs_h), n_envs_w(n_envs_w), rgb({n_envs_h, height, n_envs_w, width, 3}), depth({n_envs_h, height, n_envs_w, width}), nearest_pt({n_envs_h * n_envs_w, 3}),obstacle_pt({n_envs_h * n_envs_w, 100, 3}), n_envs(n_envs_h * n_envs_w), test(test)
    {
        int argc = 0;
        glutInit(&argc, nullptr);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowPosition(0, 0);
        if (test)
            glutInitWindowSize(1000, 2000);
        else
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

    void set_obstacles(py::array_t<float_t> drone_p)
    {
        auto p=drone_p.unchecked<2>();
        envs.clear();
        envs.resize(n_envs);
        for (int i = 0; i < n_envs/4; i++)
        {
            Geometry *d;
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    d = new Cone(0.5, 0.5);
                    int q = i + j * n_envs / 4, q1 = i + k * n_envs / 4;
                    if (q == q1)
                        envs[q].emplace_back(d, 
                            Vector3f{-10.0,-10.0,-5.0},
                            Vector3f{0.0,0.0,0.0});
                    else
                        envs[q].emplace_back(d, 
                            Vector3f{p(q1,0), p(q1,1), p(q1,2)},
                            Vector3f{0.0,0.0,0.0});
                }
            }
            int n_obstacles = (rd() % 30) + 11;
            for (int j = 4; j < n_obstacles; j++)
            {
                float x = float(rd()) / rd.max() * 18 + 4;
                float y = float(rd()) / rd.max() * 16 - 8;
                float z = float(rd()) / rd.max() * 6 - 1;
                float vx = 0, vy = 0, vz = 0;
                if (float(rd()) / rd.max() < 0.5 && !test) {
                    vx = float(rd()) / rd.max() * 2 - 1;
                    vy = float(rd()) / rd.max() * 2 - 1;
                    vz = float(rd()) / rd.max() * 2 - 1;
                }
                float r = -logf(float(rd()) / rd.max());
                for (int k = 0; k < 4; k++)
                {
                    Geometry *m;
                    switch (rd() % 4)
                    {
                    case 0:
                        m = new Cube(r + 0.1);
                        break;
                    case 1:
                        m = new Ball(r + 0.1);
                        break;

                    default:
                        m = new Cone(r / 4 + 0.1, 100);
                        z = -1;
                        vz = 0;
                        break;
                    }
                    int p = i + k*n_envs/4;
                    envs[p].emplace_back(m, Vector3f{x, y, z}, Vector3f{vx, vy, vz});
                }
            }
        }
    };

    std::tuple<py::array_t<uint8_t>, py::array_t<float_t>, py::array_t<float_t>, py::array_t<float_t>> render(
        py::array_t<float_t> cameras, float ctl_dt, py::array_t<float_t> drone_p, bool flush)
    {
        int height = depth.shape(1);
        int width = depth.shape(3);
        assert(cameras.shape(0) == n_envs);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        auto r = cameras.unchecked<2>();
        auto drone_p1 = drone_p.unchecked<2>();
        auto nearest_pt_ptr = nearest_pt.mutable_unchecked<2>();
        auto obstacle_pt_ptr = obstacle_pt.mutable_unchecked<3>();

        auto n_envs = envs.size();
        for (int i = 0; i < cameras.shape(0); i++)
        {
            glLoadIdentity();
            glViewport(width * (i % n_envs_w), height * (i / n_envs_w), width, height);
            gluPerspective(180 * 0.354, 12. / 9, 0.01f, 10.0f);

            Vector3f camera_p{r(i, 0), r(i, 1), r(i, 2)};
            Vector3f camera_f{r(i, 3), r(i, 4), r(i, 5)};
            Vector3f camera_u{r(i, 6), r(i, 7), r(i, 8)};

            set_camera(camera_p, camera_f, camera_u);

            nearest_pt_ptr(i, 0) = camera_p.x;
            nearest_pt_ptr(i, 1) = camera_p.y;
            nearest_pt_ptr(i, 2) = fmin(-1, camera_p.z);
            float nearest_distance = fabs(camera_p.z - fmin(-1, camera_p.z));

            for (int j=0;j<envs[i].size();j++)
            {
                auto &ball = envs[i][j];
                if (j<4 and j!=i/n_envs) 
                    ball.set_p(Vector3f{drone_p1(i%(n_envs/4)+j*n_envs/4,0),drone_p1(i%(n_envs/4)+j*n_envs/4,1),drone_p1(i%(n_envs/4)+j*n_envs/4,2)});
                else ball.update(ctl_dt);
                Vector3f p = ball.get_p();
                Vector3f n_pt = ball.nearestPt(camera_p);
                Vector3f cam2pt = n_pt - camera_p;
                float forward_distance = cam2pt.dot(camera_f);
                float distance = cam2pt.norm();
                obstacle_pt_ptr(i, j, 0) = p.x;
                obstacle_pt_ptr(i, j, 1) = p.y;
                obstacle_pt_ptr(i, j, 2) = p.z;
                if (-1 < forward_distance && forward_distance < 10 && distance < 10) {
                    if (distance < nearest_distance)
                    {
                        nearest_pt_ptr(i, 0) = n_pt.x;
                        nearest_pt_ptr(i, 1) = n_pt.y;
                        nearest_pt_ptr(i, 2) = n_pt.z;
                        nearest_distance = distance;
                    }
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
        if (test) {
            glLoadIdentity();
            glViewport(width*2, height*2, 1000 - width*2, 2000 - height*2);
            gluPerspective(180 * 0.354 * 0.8, 9. / 25, 0.01f, 300.0f);

            //gluLookAt(4,30,30,20,-30,-30,0,0,1);
            gluLookAt(-30,0,50,60,0,-50,0,0,1);
            
            for (int j=0;j<envs[0].size();j++)
            {
                auto &ball = envs[0][j];
                if (j==0) ball.set_p(Vector3f{drone_p1(0,0),drone_p1(0,1),drone_p1(0,2)});
                ball.draw();
                if (j==0) ball.set_p(Vector3f{-10,-10,-5});
                }
        }
        if (flush)
            glFlush();
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width * n_envs_w, height * n_envs_h, GL_BGR, GL_UNSIGNED_BYTE, rgb.request().ptr);
        glReadPixels(0, 0, width * n_envs_w, height * n_envs_h, GL_DEPTH_COMPONENT, GL_FLOAT, depth.request().ptr);
        return {rgb, depth, nearest_pt, obstacle_pt};
    };
    ~Env(){};
};
