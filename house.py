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
BRN=(60,86,132)


def spew(fname, contents):
    f = open(fname, "w")
    f.write(contents)
    f.close()

def irnd(x):
   return int(np.round(x))

def color_of(t):
   if t == 'X': return RED
   if t == 'Y': return GRN
   if t == 'Z': return BLU
   if t == 'D': return YLW

def pt(img, x, y=None, scl=1.0):
    h,w = img.shape[0:2]
    col = x
    row = y
    if row is None:
        if x.shape[0]>2:
            col = x[0,0] / x[2,0]
            row = x[1,0] / x[2,0]
        elif x.shape[0]==2:
            col = x[0,0]
            row = x[1,0]

    if row is None:
       stophere=1

    dx = col - w//2
    dy = row - h//2
    dx *= scl
    dy *= scl

    return (irnd(w//2 + dx), irnd(h//2 - dy))

def mark_origin(img, cam):
    c = cam @ np.array([[0],[0],[0],[1]])
    cv2.circle(img, pt(img,c), 3, (255,255,255))

def project_gline(img, cam, gp0, gp1, color, wid=2, scl=1.0):
    ip0 = cam @ gp0
    ip1 = cam @ gp1
    cv2.line(img, pt(img,ip0,scl=scl), pt(img,ip1,scl=scl), color, wid)



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


def scale_ip(img, x, y, s, flipy=False):
   h,w = img.shape[0:2]
   dx = x - w//2
   dy = y - h//2
   if flipy:
      return (w//2 + dx*s, h//2 - dy*s)
   else:
      return (w//2 + dx*s, h//2 + dy*s)


class Primitive():
   def __init__(s, typ, color):
      s.col = color
      s.typ = typ

   def draw(s, img, cam, wid=1, extend=False, scl=1.0):
      pass


class ImgPoly(Primitive):
   def __init__(s, points, color):
      super().__init__('IPOLY', color)
      s.verts = points

   def draw(s, img, cam, wid=1, extend=False, scl=1.0):
      sverts = []
      for v in s.verts:
         sverts.append(scale_ip(img, v[0], v[1], scl))
      vs = [np.array(sverts, np.int32).reshape((-1,1,2))]
      cv2.fillPoly(img, vs, s.col)


class ImgCircle(Primitive):
   def __init__(s, center, radius, color):
      super().__init__('CIRCLE', color)
      s.ctr = center
      s.rad = radius

   def draw(s, img, cam, wid=1, extend=False, scl=1.0):
      col,row = scale_ip(img, s.ctr[0], s.ctr[1], scl, flipy=True)
      # scale the radius?
      cv2.circle(img, (irnd(col), irnd(row)), s.rad, s.col, -1)


class Line(Primitive):
   def __init__(s, gpt0, gpt1, color, typ):
      super().__init__(typ, color)
      s.gp0 = gpt0
      s.gp1 = gpt1
      s.vp  = None # fill in vanishing point when camera is known

   def draw(s, img, cam, wid=1, extend=False, scl=1.0):
      # House line may be thicker width
      project_gline(img, cam, s.gp0, s.gp1, s.col, wid=wid, scl=scl)

      # Extended line toward vanishing point is always wid=1
      if extend > 0 and s.typ != 'D':
         ip3 = cam @ s.gp1
         ip2 = ip3[0:2, :] / ip3[2, 0]
         vp = s.vp
         ept = extend * vp + (1 - extend) * ip2

         ip2 = np.array(scale_ip(img, ip2[0,0], ip2[1,0], scl)).reshape((2,1))
         ept = np.array(scale_ip(img, ept[0,0], ept[1,0], scl)).reshape((2,1))

         cv2.line(img, pt(img, ip2), pt(img, ept), s.col, 1)



class HouseWriter():
    def __init__(s, fname='house.avi', width=1920, height=1080):
       s.imgw = width
       s.imgh = height
       fourcc = cv2.VideoWriter_fourcc(*'XVID')
       s.writer = cv2.VideoWriter(fname, fourcc, 15, (width,height))

       s.foc = 200
       s.ppx = s.imgw//2
       s.ppy = s.imgh//2
       s.K = buildK(s.foc, s.ppx, s.ppy)

       s.azi = 25
       s.tlt = 70
       s.swg = 180
       s.tx = 0
       s.ty = -.9
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
       s.gprf = np.array([[.5], [1.5], [.5], [1]])
       s.gprb = np.array([[.5], [1.5], [.5], [1]])

       # build the primitives out of the vertices
       # layered back to front so it looks good
       s.prims = []

       # background rectangles
       b = 10000
       s.prims.append(ImgPoly([[-b,-b],[b,-b],[b,b],[-b,b]],BLK))
       w=s.imgw
       h=s.imgh
       s.prims.append(ImgPoly([[0,0],[w,0],[w,h],[0,h]],WHT))
       # back wall and roof
       s.prims.append(Line(s.gp001,s.gp101,RED,'X'))
       s.prims.append(Line(s.gp011,s.gp111,RED,'X'))
       s.prims.append(Line(s.gp011,s.gp001,GRN,'Y'))
       s.prims.append(Line(s.gp111,s.gp101,GRN,'Y'))
       # right wall
       s.prims.append(Line(s.gp100,s.gp101,BLU,'Z'))
       s.prims.append(Line(s.gp110,s.gp111,BLU,'Z'))
       s.prims.append(Line(s.gp110,s.gp100,GRN,'Y'))
       # left wall
       s.prims.append(Line(s.gp000,s.gp001,BLU,'Z'))
       s.prims.append(Line(s.gp010,s.gp011,BLU,'Z'))
       s.prims.append(Line(s.gp010,s.gp000,GRN,'Y'))
       # front
       s.prims.append(Line(s.gp000,s.gp100,RED,'X'))
       s.prims.append(Line(s.gp010,s.gp110,RED,'X'))
       # roof
       s.prims.append(Line(s.gp011,s.gprb, YLW,'D'))
       s.prims.append(Line(s.gp111,s.gprb, YLW,'D'))
       s.prims.append(Line(s.gp010,s.gprf, YLW,'D'))
       s.prims.append(Line(s.gp110,s.gprf, YLW,'D'))
       #s.prims.append(Line(s.gprb, s.gprf, BLU,'Z'))

       s.grounds = []
       lo=-1
       hi=2
       for xz in range(lo,hi+1):
          s.grounds.append(Line(np.array([[xz],[0],[lo],[1]]),
                                     np.array([[xz],[0],[100],[1]]),BRN,'G'))
          s.grounds.append(Line(np.array([[lo],[0],[xz],[1]]),
                                     np.array([[100],[0],[xz],[1]]),BRN,'G'))
       # this is not workin out
       s.grounds = []

       s.vps = {}
       s.updateVP()


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

    def project(s, gp, flip=False):
       ip = s.cam @ gp
       ip *= 1.0/ip[2,0]
       ip[1,0] = s.imgh - ip[1,0] if flip else ip[1,0]
       return (ip[0,0], ip[1,0])

    def updateVP(s):
       for t in ['X','Y','Z']:
          # Find two primitives of type t
          p0 = p1 = None
          for p in s.prims:
             if p.typ == t:
                if   p0 is None: p0 = p
                elif p1 is None: p1 = p
          # VP is intersection of p0 and p1 (as projected)
          ip00  = s.project(p0.gp0)
          ip01  = s.project(p0.gp1)
          rise0 = ip01[1]-ip00[1]
          run0  = ip01[0]-ip00[0]
          ip10  = s.project(p1.gp0)
          ip11  = s.project(p1.gp1)
          rise1 = ip11[1] - ip10[1]
          run1  = ip11[0] - ip10[0]

          A = np.array([[rise0, -run0],
                        [rise1, -run1]])
          rhs = np.array([[rise0*ip00[0] - run0*ip00[1]],
                          [rise1*ip10[0] - run1*ip10[1]]])
          vp = np.linalg.solve(A, rhs)
          s.vps[t] = ImgCircle(vp.reshape(2).tolist(), 5, color_of(t))

          for p in s.prims:
             if p.typ==t:
                p.vp = vp

    def house(s, image_in=None, fat=None, extend=0, scale=1.0,
              drawVPs=False, drawVT=False,
              azim=None, tilt=None, swng=None,
              tx=None, ty=None, tz=None, f=None,
              dump=False, show=False, write=True):
       if image_in is None:
           img = np.zeros((s.imgh, s.imgw, 3), np.uint8)
       else:
           img = image_in
       s.updateK(f, None, None)
       s.updateRt(azim, tilt, swng, tx, ty, tz)
       s.cam = s.K @ s.Rt
       if extend:
          s.updateVP()

       for p in s.prims:
          w = 4 if fat is not None and fat==p.typ else 2
          p.draw(img, s.cam, w, extend, scale)

       if drawVPs:
          for vp in s.vps.values():
             vp.draw(img, s.cam, w, extend, scale)

       #mark_origin(img, s.cam)
       if dump:   cv2.imwrite('dbg.png', img)
       if show:   cv2.imshow('House', img)
       if write:  s.writer.write(img)

       print('.', end='')



       
    def __del__(s):
        s.writer.release()

    def __enter__(s):
        return s

    def __exit__(s, exc_type, exc_value, exc_traceback):
        s.writer.release()
        return True



hw = HouseWriter(width=720, height=480)

hw.updateRt()


print('Still',end='')
for a in range(30):
   hw.house()

print('\nFats',end='')
for fat_ax in ['X','Z','Y','D']:
   for a in range(15):
      hw.house(fat=fat_ax)

print('\nExtend',end='')
for e in np.arange(0.0, 1.01, 0.05):
   hw.house(extend=e)

print('\nShrink', end='')
for s in np.arange(1.0, 0.3, -0.05):
   hw.house(extend=1, scale=s, drawVPs=True)







hw.writer.release()
