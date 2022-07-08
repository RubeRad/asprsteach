#! /usr/bin/python3

import math  as m
import numpy as np
import cv2

RED=(0,0,255)
GRN=(0,255,0)
BLU=(255,0,0)
YLW=(0,255,255)
WHT=(255,255,255)
BLK=(0,0,0)



def spew(fname, contents):
    f = open(fname, "w")
    f.write(contents)
    f.close()

def pt(img, x, y=None):
    if y is None:
        if x.shape[0]>2:
            x[0,0] /= x[2,0]
            x[1,0] /= x[2,0]
        return (int(np.round(x[0,0])), img.shape[0]-int(np.round(x[1,0])))
    return (int(np.round(x)), img.shape[0]-int(np.round(y)))

def mark_origin(img, cam):
    c = cam @ np.array([[0],[0],[0],[1]])
    cv2.circle(img, pt(img,c), 3, (255,255,255))

def project_gline(img, cam, gp0, gp1, color, fname='dbg.png'):
    ip0 = cam @ gp0
    ip1 = cam @ gp1

    cv2.line(img, pt(img,ip0), pt(img,ip1), color, 4)
    cv2.imwrite(fname, img)


def build_Rt(az, ti, sw, cx, cy, cz):
    a = m.radians(az)     # azimuth=0 --> north
    t = m.radians(90- ti) # tilt=90   --> horizontal pitch
    s = m.radians(sw-180) # swing=180 --> top up
    Ro = np.eye(3)
    Ra = np.array([[ m.cos(a),    0,    -m.sin(a)],
                   [ 0,           1,     0       ],
                   [ m.sin(a),    0,     m.cos(a)]])
    Rt = np.array([[ 1,  0,        0       ],
                   [ 0,  m.cos(t), m.sin(t)],
                   [ 0, -m.sin(t), m.cos(t)]])
    Rs = np.array([[ m.cos(s), m.sin(s), 0],
                   [-m.sin(s), m.cos(s), 0],
                   [ 0,        0,        1]])
    R = Rs @ Rt @ Ra @ Ro
    t = np.array([[cx],[cy],[cz]])
    # TBD need to rotate cx, cy, cz
    return(np.hstack((R,t)))


    

imgw = 720
imgh = int(imgw*3/4) # 4:3 ratio
f  = 1000
cx = imgw/2
cy = imgh/2
K  = np.array([[f, 0, cx],
               [0, f, cy],
               [0, 0,  1]])


# 8 vertices
gp000 = np.array([[0], [0], [0], [1]])
gp001 = np.array([[0], [0], [1], [1]])
gp010 = np.array([[0], [1], [0], [1]])
gp011 = np.array([[0], [1], [1], [1]])
gp100 = np.array([[1], [0], [0], [1]])
gp101 = np.array([[1], [0], [1], [1]])
gp110 = np.array([[1], [1], [0], [1]])
gp111 = np.array([[1], [1], [1], [1]])
gprf  = np.array([[.5], [1.5], [0], [1]])
gprb  = np.array([[.5], [1.5], [1], [1]])

vvv = True

for i in range(10):
    img = np.zeros((imgh,imgw,3), np.uint8)
    Rt = build_Rt(i*10, 90, 180, 0, -1, 5)
    KRt = K@Rt
    fname = 'cube_t090_a' + str(i*10) + '.png'
    tname = fname + ".txt"
    
    if (vvv):
        print(fname)

    txt = '{"image_fname": "'+fname+'",\n"vlines": [\n'

    # X lines are red
    project_gline(img, KRt, gp000, gp100, RED)
    project_gline(img, KRt, gp001, gp101, RED)
    project_gline(img, KRt, gp010, gp110, RED)
    project_gline(img, KRt, gp011, gp111, RED)
    # Y lines are green
    project_gline(img, KRt, gp000, gp010, GRN)
    project_gline(img, KRt, gp001, gp011, GRN)
    project_gline(img, KRt, gp100, gp110, GRN)
    project_gline(img, KRt, gp101, gp111, GRN)
    # Z lines are blue
    project_gline(img, KRt, gp000, gp001, BLU)
    project_gline(img, KRt, gp010, gp011, BLU)
    project_gline(img, KRt, gp100, gp101, BLU)
    project_gline(img, KRt, gp110, gp111, BLU)

    # Let's add a roof to make a house!
    project_gline(img, KRt, gprf, gprb, BLU)  # Top roofline is Z=BLU
    # diagonals are not main axes so let's make them yellow
    project_gline(img, KRt, gprf, gp010, YLW)
    project_gline(img, KRt, gprf, gp110, YLW)
    project_gline(img, KRt, gprb, gp011, YLW)
    project_gline(img, KRt, gprb, gp111, YLW)

    mark_origin(img, KRt)


    cv2.imwrite(fname, img)
