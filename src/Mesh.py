import numpy as np
from OpenGL.GL import *


def read_obj(filename):
    v = []
    vt = []
    vn = []
    vertices = []

    with open(filename, "r") as file:
        line = file.readline()

        while line:
            words = line.split(" ")

            # coordenadas dos vertices
            if words[0] == "v":
                vertex = [
                    float(words[1]),
                    float(words[2]),
                    float(words[3])
                ]
                v.append(vertex)

            # coordenadas de textura
            if words[0] == "vt":
                vertex_t = [
                    float(words[1]),
                    float(words[2])
                ]
                vt.append(vertex_t)

            # normais do vertice
            if words[0] == "vn":
                vertex_n = [
                    float(words[1]),
                    float(words[2]),
                    float(words[3])
                ]
                vn.append(vertex_n)

            # faces do objeto
            if words[0] == "f":
                for i in range(len(words) - 3):
                    read_face(words[1], v, vt, vn, vertices)
                    read_face(words[2 + i], v, vt, vn, vertices)
                    read_face(words[3 + i], v, vt, vn, vertices)

            line = file.readline()

    return vertices

def read_face(face_description, v, vt, vn, vertices):
    v_vt_vn = face_description.split("/")
    
    for element in v[int(v_vt_vn[0]) - 1]:
        vertices.append(element)
    for element in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(element)
    for element in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(element)

class Mesh:
    def __init__(self, filename):

        # x, y, z, s, t, nx, ny, nz
        vertices = read_obj(filename)
        self.vertex_count = len(vertices)//8
        vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        #Vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def arm_for_drawing(self):
        glBindVertexArray(self.vao)
    
    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))
