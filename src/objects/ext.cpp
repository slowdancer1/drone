#include "ext.hpp"

using namespace std;

void ExtShape::readmtl(string mtl_file) {
    string line;
    stringstream mtl_file_stream;
    mtl_file_stream << mtl_file;
    int mtl_index = 0;
    mtlSets.push_back(vector<float>(3, 1.0f));
    while (!mtl_file_stream.eof()) {
        getline(mtl_file_stream, line);
        vector<string> parameters;
        string tailMark = " ";
        string ans = "";

        line = line.append(tailMark);
        for (unsigned int i = 0; i < line.length(); i++) {
            char ch = line[i];
            if (ch != ' ')  ans += ch;
            else parameters.push_back(ans),ans = "";
        }

        if (parameters[0] == "newmtl") {
            mtl_index++;
            mtlMap.insert(pair<string, int>(parameters[1], mtl_index));
        } else if (parameters.size() == 4 && parameters[0] == "Kd") {
            vector<float> RGB;
            for (int i = 1; i < 4; i++) {
                RGB.push_back(atof(parameters[i].c_str()));
            }
            mtlSets.push_back(RGB);
        }
    }
}

void ExtShape::readobj(string obj_file) {
    /****************************
     * Reference: https://chenjiayang.me/2016/12/07/OpenGL/
     * Date: December 7, 2016
     * Author: ChenJY
     ****************************/
    
    string line;
    stringstream obj_file_stream;
    obj_file_stream << obj_file;
    int mtl_index = 0;
    while (!obj_file_stream.eof()) {
        getline(obj_file_stream, line);
        vector<string> parameters;
        string tailMark = " ";
        string ans = "";

        line = line.append(tailMark);
        for (unsigned int i = 0; i < line.length(); i++) {
            char ch = line[i];
            if (ch != ' ')  ans += ch;
            else parameters.push_back(ans),ans = "";
        }
        if (parameters[0] == "usemtl") {
            mtl_index = mtlMap[parameters[1]];
        }
        if (parameters.size() == 4) {
            if (parameters[0] == "v") {
                vector<float> Point;
                for (int i = 1; i < 4; i++) {
                    float xyz = atof(parameters[i].c_str());
                    Point.push_back(xyz);
                }
                vSets.push_back(Point);
            }

            else if (parameters[0] == "f") {
                vector<int> vIndexSets;
                for (int i = 1; i < 4; i++) {
                    string x = parameters[i];
                    string ans = "";
                    for (unsigned int j = 0; j < x.length(); j++) {
                        char ch = x[j];
                        if (ch != '/') ans += ch;
                        else break;
                    }
                    int index = atof(ans.c_str());
                    vIndexSets.push_back(index - 1);
                }
                vIndexSets.push_back(mtl_index);
                fSets.push_back(vIndexSets);
            }
        }
    }
}

ExtShape::ExtShape(string obj_file, string mtl_file) {
    readmtl(mtl_file);
    readobj(obj_file);
}

void ExtShape::draw() {
    // glRotatef(90, 1, 0, 0);
    glBegin(GL_TRIANGLES);
    for (unsigned int i = 0; i < fSets.size(); i++) {
        float SV1[3];
        float SV2[3];
        float SV3[3];
        float mtl[3];

        // Get Index
        int firstVertexIndex = (fSets[i])[0];
        int secondVertexIndex = (fSets[i])[1];
        int thirdVertexIndex = (fSets[i])[2];
        int materialIndex = (fSets[i])[3];

        // Get Value
        SV1[0] = (vSets[firstVertexIndex])[0];
        SV1[1] = (vSets[firstVertexIndex])[1];
        SV1[2] = (vSets[firstVertexIndex])[2];

        SV2[0] = (vSets[secondVertexIndex])[0];
        SV2[1] = (vSets[secondVertexIndex])[1];
        SV2[2] = (vSets[secondVertexIndex])[2];

        SV3[0] = (vSets[thirdVertexIndex])[0];
        SV3[1] = (vSets[thirdVertexIndex])[1];
        SV3[2] = (vSets[thirdVertexIndex])[2];

        mtl[0] = (mtlSets[materialIndex])[0];
        mtl[1] = (mtlSets[materialIndex])[1];
        mtl[2] = (mtlSets[materialIndex])[2];

        // Draw
        glColor4f(mtl[0], mtl[1], mtl[2], 1.0f);
        glVertex3f(SV1[0], SV1[1], SV1[2]);
        glVertex3f(SV3[0], SV3[1], SV3[2]);
        glVertex3f(SV2[0], SV2[1], SV2[2]);
    }
    glEnd();
}
