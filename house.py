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


def buildK(f, cx, cy):
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


def buildRt(az, ti, sw, tx, ty, tz):
    a = m.radians(az)     # azimuth=0 --> north
    t = m.radians(90- ti) # tilt=90   --> horizontal pitch
    s = m.radians(sw-180) # swing=180 --> top up
    Ra = np.array([[ m.cos(a),    0,    -m.sin(a)],
                   [ 0,           1,     0       ],
                   [ m.sin(a),    0,     m.cos(a)]])
    Rt = np.array([[ 1,  0,        0       ],
                   [ 0,  m.cos(t), m.sin(t)],
                   [ 0, -m.sin(t), m.cos(t)]])
    Rs = np.array([[ m.cos(s), m.sin(s), 0],
                   [-m.sin(s), m.cos(s), 0],
                   [ 0,        0,        1]])
    R = Rs @ Rt @ Ra
    t = np.array([[tx],[ty],[tz]])
    return(np.hstack((R,t)))



class HouseWriter():
    def __init__(s, fname='house.avi', width=1920, height=1080):
       s.imgw = width
       s.imgh = height
       fourcc = cv2.VideoWriter_fourcc(*'XVID')
       s.writer = cv2.VideoWriter(fname, fourcc, 15, (width,height))

       s.foc = 100
       s.ppx = s.imgw//2
       s.ppy = s.imgh//2
       s.K = buildK(s.foc, s.ppx, s.ppy)

       s.azi = 10
       s.tlt = 90
       s.swg = 180
       s.tx = 0
       s.ty = -1
       s.tz = 1
       s.Rt = buildRt(s.azi, s.tlt, s.swg, s.tx, s.ty, s.tz)
       
       s.cam = s.K @ s.Rt
       
       # 8 vertices
       s.gp000 = np.array([[0], [0], [0], [1]])
       s.gp001 = np.array([[0], [0], [1], [1]])
       s.gp010 = np.array([[0], [1], [0], [1]])
       s.gp011 = np.array([[0], [1], [1], [1]])
       s.gp100 = np.array([[1], [0], [0], [1]])
       s.gp101 = np.array([[1], [0], [1], [1]])
       s.gp110 = np.array([[1], [1], [0], [1]])
       s.gp111 = np.array([[1], [1], [1], [1]])
       s.gprf = np.array([[.5], [1.5], [0], [1]])
       s.gprb = np.array([[.5], [1.5], [1], [1]])

       # 12 lines for the cube (and 5 more for the roof)
       s.xlines = [(s.gp000, s.gp100), (s.gp001, s.gp101), (s.gp010, s.gp110), (s.gp011, s.gp111)]
       s.ylines = [(s.gp000, s.gp010), (s.gp001, s.gp011), (s.gp100, s.gp110), (s.gp101, s.gp111)]
       s.zlines = [(s.gp000, s.gp001), (s.gp010, s.gp011), (s.gp100, s.gp101), (s.gp110, s.gp111), (s.gprf, s.gprb)]
       s.dlines = [(s.gprf, s.gp010), (s.gprf, s.gp110), (s.gprb, s.gp011), (s.gprb, s.gp111)]


    def updateK(s, f, ppx, ppy):
       focal = f   if f   is not None else s.foc; s.foc = focal
       ctrx  = ppx if ppx is not None else s.ppx; s.ppx = ctrx
       ctry  = ppy if ppy is not None else s.ppy; s.ppy = ctry
       s.K = buildK(focal, ctrx, ctry)
       return s.K
    
    def updateRt(s, a=None, t=None, w=None, x=None, y=None, z=None):
       azim = a if a is not None else s.azi; s.azi = azim
       tilt = t if t is not None else s.tlt; s.tlt = tilt
       swng = w if w is not None else s.swg; s.swg = swng
       posx = x if x is not None else s.tx;  s.tx  = posx
       posy = y if y is not None else s.ty;  s.ty  = posy
       posz = z if z is not None else s.tz;  s.tz  = posz
       s.Rt = buildRt(azim, tilt, swng, posx, posy, posz)
       return s.Rt
        

    def house(s, image_in=None,
              azim=None, tilt=None, swng=None,
              tx=None, ty=None, tz=None, f=None,
              dump=False, show=False, write=False):
       if image_in is None:
           img = np.zeros((s.imgh, s.imgw, 3), np.uint8)
       else:
           img = image_in
       s.updateK(f, None, None)
       s.updateRt(azim, tilt, swng, tx, ty, tz)
       s.cam = s.K @ s.Rt

       for xx in s.xlines: project_gline(img, s.cam, xx[0], xx[1], RED)
       for yy in s.ylines: project_gline(img, s.cam, yy[0], yy[1], GRN)
       for zz in s.zlines: project_gline(img, s.cam, zz[0], zz[1], BLU)
       for dd in s.dlines: project_gline(img, s.cam, dd[0], dd[1], YLW)

       mark_origin(img, s.cam)
       if dump:   cv2.imwrite('dbg.png', img)
       if show:   cv2.imshow('House', img)
       if write:  s.writer.write(img)



       
    def __del__(s):
        s.writer.release()

    def __enter__(s):
        return s

    def __exit__(s, exc_type, exc_value, exc_traceback):
        s.writer.release()
        return True



with HouseWriter(width=720, height=480) as hw:
    print('A',end='')
    for a in range(40):
        print('.',end='')
        hw.house(azim=a, write=True)

    print('\nT',end='')
    for t in range(90,60,-1):
        print('.',end='')
        hw.house(tilt=t, write=True)

    print('\nZ',end='')
    for zi in range(20):
        print('.',end='')
        hw.house(tz=1+zi/10, write=True)

    print('\nF',end='')
    for f in range(100,50,-10):
        print('.',end='')
        hw.house(f=f, write=True)