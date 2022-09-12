#! /usr/bin/python3

import copy
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
PPL=(200,0,200)
SKY=(235,206,135)


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

def D2to3(xyw): # maybe actually 3 to 4
   if len(xyw)==2:       # 2tuple
      x,y = xyw
      return np.array([[x],   [y],   [0], [1]])
   else:                 # numpy homomorphic 3x1
      x,y,w = xyw.reshape(3).tolist()
      return np.array([[x/w], [y/w], [0], [1]])



# convert a numpy column-vector into a 2-tuple for cv2 drawing functions
# (possibly a homomorphic 3-vector)
def pt(img, x, y=None, scl=1.0, flipy=True):
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

    dx = col - w//2
    dy = row - h//2
    dx *= scl
    dy *= scl
    if flipy: dy *= -1

    return (irnd(w//2 + dx), irnd(h//2 + dy))


# convert a 2-tuple or 3-tuple into a numpy column vector
# (possibly concatenating a homomorphic 1)
def npt(tup, hom=False):
   l = len(tup)
   if hom: l += 1
   pt = np.zeros((l,1))
   for i in range(len(tup)):
      pt[i,0] = tup[i]
   if hom:
      pt[l,0] = 1
   return pt


def mark_origin(img, cam):
    c = cam @ npt((0,0,0,1))
    cv2.circle(img, pt(img,c), 3, (255,255,255))

def project_gline(img, cam, gp0, gp1, color, wid=2, scl=1.0):
    ip0 = cam @ gp0   # project into the image
    ip1 = cam @ gp1
    #print('ip1 is',ip1)
    ip00 = pt(img,ip0,scl=scl) # take care of homomorphism, yflip, scale, and rounding
    ip11 = pt(img,ip1,scl=scl)
    #print('ip11 is',ip11)
    cv2.line(img, ip00, ip11, color, wid)



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
    t = npt((tx,ty,tz))
    return(np.hstack((R,t)))


def scale_ip(img, x_or_xy, y=None, s=1.0, flipy=False):
   h,w = img.shape[0:2]
   if y is None:
      y = x_or_xy[1]
      x = x_or_xy[0]
   else:
      x = x_or_xy
   dx = x - w//2
   dy = y - h//2
   if flipy:
      return (w//2 + dx*s, h//2 - dy*s)
   else:
      return (w//2 + dx*s, h//2 + dy*s)


# intersect lines defined by two 2-tuples each
def line_inter(ip00,ip01,  ip10,ip11):
   rise0 = ip01[1] - ip00[1]
   run0  = ip01[0] - ip00[0]
   rise1 = ip11[1] - ip10[1]
   run1  = ip11[0] - ip10[0]

   A = np.array([[rise0, -run0],
                 [rise1, -run1]])
   rhs = np.array([[rise0 * ip00[0] - run0 * ip00[1]],
                   [rise1 * ip10[0] - run1 * ip10[1]]])
   vp = np.linalg.solve(A, rhs)
   return (vp[0,0], vp[1,0])


class Primitive():
   def __init__(s, typ, color):
      s.col = color
      s.typ = typ

   def draw(s, img, cam, wid=1, extend=False, scl=1.0):
      pass

   def draw2D(s, img, scl, offx, offy, wid=4):
      pass

   def drawdraw(s, img, cam0, cam1, wid=1):
      pass

   def flipy(s, imgh):
      pass


class ImgPoly(Primitive):
   def __init__(s, points, color, filled=True):
      super().__init__('IPOLY', color)
      s.verts = points
      s.filled = filled

   def draw(s, img, cam, wid=1, extend=False, scl=1.0):
      sverts = []
      for v in s.verts:
         sverts.append(scale_ip(img, v[0], v[1], scl))
      vs = [np.array(sverts, np.int32).reshape((-1,1,2))]
      if s.filled:
         cv2.fillPoly(img, vs, s.col)
      else:
         for i in range(len(s.verts)):
            cv2.line(img, s.verts[i], s.verts[(i+1)%len(s.verts)], s.col, 2)

   def drawdraw(s, img, cam0, cam1, wid=1):
      h = img.shape[0]
      sverts = []
      for v in s.verts:
         gp = np.array([[v[0]],[h-v[1]],[0],[1]])
         ip = cam1 @ gp
         sverts.append(pt(img,ip))
      vs = [np.array(sverts, np.int32).reshape((-1, 1, 2))]
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

   def draw2D(s, img, scl, offx, offy, wid=4):
      off = np.array([offx,offy])
      ctr = np.array(s.ctr) / scl + off
      ictr = tuple(np.round(ctr).astype(int).tolist())
      cv2.circle(img, ictr, s.rad, s.col, -1)

   def drawdraw(s, img, cam0, cam1, wid=1):
      h = img.shape[0]
      c = s.ctr
      gp = np.array([[c[0]], [h-c[1]], [0], [1]])
      ip = cam1 @ gp
      cv2.circle(img, pt(img,ip), s.rad, s.col, -1)

   def flipy(s, imgh):
      x,y = s.ctr
      s.ctr = (x, imgh-y)


class ImgLine(Primitive):
   def __init__(s, ip0, ip1, color, typ):
      s.ip0 = ip0
      s.ip1 = ip1
      s.col = color
      s.typ = typ

   def drawdraw(s, img, cam0, cam1, wid=1):
      h=img.shape[0]
      gp0 = np.array([[s.ip0[0]], [h-s.ip0[1]], [0], [1]])
      gp1 = np.array([[s.ip1[0]], [h-s.ip1[1]], [0], [1]])
      ip0 = cam1 @ gp0
      ip1 = cam1 @ gp1
      cv2.line(img, pt(img,ip0), pt(img,ip1), s.col, wid)

   def flipy(s, imgh):
      x,y = s.ip0
      s.ip0 = (x,imgh-y)
      x,y = s.ip1
      s.ip1 = (x-imgh-y)


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
         vp = npt(s.vp)
         ept = extend * vp + (1 - extend) * ip2

         ip2 = npt(scale_ip(img, ip2[0,0], ip2[1,0], scl))
         ept = npt(scale_ip(img, ept[0,0], ept[1,0], scl))

         cv2.line(img, pt(img, ip2), pt(img, ept), s.col, 1)

   def draw2D(s, img, scl, offx, offy, wid=4):
      off = np.array([offx,offy])
      xy0 = np.array(s.gp0) / scl + off
      xy1 = np.array(s.gp1) / scl + off
      rc0 = tuple(np.round(xy0).astype(int).tolist())
      rc1 = tuple(np.round(xy1).astype(int).tolist())

      cv2.line(img, rc0,rc1,s.col, wid)

   def extendToward(s, pt2, ext):
      p0 = np.array(s.gp0)
      p1 = np.array(s.gp1)
      v01 = p1-p0
      l01 = np.linalg.norm(v01)

      p2 = np.array(pt2, np.float64)
      v12 = p2-p1
      l12 = np.linalg.norm(v12)
      len = l01 + ext * l12

      v02 = p2-p0
      v02 *= 1.0/np.linalg.norm(v02)
      s.gp1 = tuple(p0 + len*v02)

   def drawdraw(s, img, cam0, cam1, wid=1):
      ip0 = cam0 @ s.gp0
      ip1 = cam0 @ s.gp1
      tmp0 = D2to3(ip0)
      tmp1 = D2to3(ip1)
      ip00 = cam1 @ tmp0
      ip11 = cam1 @ tmp1
      pt0 = pt(img,ip00)
      pt1 = pt(img,ip11)
      cv2.line(img, pt(img,ip00), pt(img,ip11), s.col, wid)



class HouseWriter():
    def __init__(s, fname='house.avi', width=1920, height=1080):
       s.imgw = width
       s.imgh = height
       fourcc = cv2.VideoWriter_fourcc(*'XVID')
       s.writer = cv2.VideoWriter(fname, fourcc, 15, (width,height))

       s.foc = 500
       s.ppx = s.imgw//2
       s.ppy = s.imgh//2
       s.K = buildK(s.foc, s.ppx, s.ppy)

       s.azi = 25
       s.tlt = 60
       s.swg = 180
       s.tx = 0
       s.ty = -.9
       s.tz = 1.0
       s.Rt = buildRt(s.azi, s.tlt, s.swg, s.tx, s.ty, s.tz)
       
       s.cam = s.K @ s.Rt
       
       # 8 vertices
       s.gp000 = npt((0,0,0,1))
       s.gp001 = npt((0,0,1,1))
       s.gp010 = npt((0,1,0,1))
       s.gp011 = npt((0,1,1,1))
       s.gp100 = npt((1,0,0,1))
       s.gp101 = npt((1,0,1,1))
       s.gp110 = npt((1,1,0,1))
       s.gp111 = npt((1,1,1,1))
       s.gprf = npt((.5,1.5,.5,1))
       s.gprb = npt((.5,1.5,.5,1))

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

       s.vps = {}
       s.updateVP()

       # "Views from Rome", 1750-78, Battista Piranesi
       # Download 1959x1334 image as piranesi.jpg from:
       # https://www.metmuseum.org/art/collection/search/406668
       s.pscl=2.2
       big = cv2.imread('piranesi.jpg')
       ph, pw = big.shape[0:2]
       s.pimg = cv2.resize(big, (int(pw / s.pscl), int(ph / s.pscl)))


    def updateK(s, f, ppx, ppy):
       focal = f   if f   is not None else s.foc; s.foc = focal
       ctrx  = ppx if ppx is not None else s.ppx; s.ppx = ctrx
       ctry  = ppy if ppy is not None else s.ppy; s.ppy = ctry
       s.K = buildK(focal, ctrx, ctry)
       s.cam = s.K @ s.Rt
       s.updateVP()
       return s.K
    
    def updateRt(s, a=None, t=None, w=None, x=None, y=None, z=None):
       azim = a if a is not None else s.azi; s.azi = azim
       tilt = t if t is not None else s.tlt; s.tlt = tilt
       swng = w if w is not None else s.swg; s.swg = swng
       posx = x if x is not None else s.tx;  s.tx  = posx
       posy = y if y is not None else s.ty;  s.ty  = posy
       posz = z if z is not None else s.tz;  s.tz  = posz
       s.Rt = buildRt(azim, tilt, swng, posx, posy, posz)
       s.cam = s.K @ s.Rt
       s.updateVP()
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
          vp = line_inter(s.project(p0.gp0), s.project(p0.gp1),
                          s.project(p1.gp0), s.project(p1.gp1))
          s.vps[t] = ImgCircle(vp, 10, color_of(t))

          for p in s.prims:
             if p.typ==t:
                p.vp = vp

    def piranesi(s, picture=1.0, extend=1.0, wid=4, caption=False):

       img = np.zeros((s.imgh,    s.imgw, 3), np.uint8)

       ph,pw = s.pimg.shape[0:2]
       oh = (s.imgh - ph) // 2
       ow = (s.imgw - pw) // 2
       if picture:
          fade = s.pimg.astype(np.float32)*picture
          img[oh:oh+ph,ow:ow+pw] = fade.astype(np.uint8)

       red1 = Line((1017,343),(1488,483),RED,'X')
       red2 = Line((166,720),(382,739),RED,'X')
       red3 = Line((637,995),(740,992),RED,'X')
       grn1 = Line((620,731),(442,783),GRN,'Y')
       grn2 = Line((1702,681),(1204,757),GRN,'Y')
       grn3 = Line((1641,1018),(1255,996),GRN,'Y')
       vpr = (2900,850)#line_inter(red1.gp0,red1.gp1, red3.gp0,red3.gp1)
       vpg = (-190,958)#line_inter(grn1.gp0,grn1.gp1, grn3.gp0,grn3.gp1)

       if extend==1.0:
          ImgCircle(vpr,10,RED).draw2D(img,s.pscl,ow,oh)
          ImgCircle(vpg,10,GRN).draw2D(img,s.pscl,ow,oh)

       if wid:
          for l in [red1,red2,red3,grn1,grn2,grn3]:
             if l.typ == 'X': l.extendToward(vpr,extend)
             else:            l.extendToward(vpg,extend)
             l.draw2D(img,s.pscl,ow,oh,wid)

       if caption:
          cv2.putText(img, 'Piranesi, "Views of Rome", metmuseum.org/art/collection/search/406668'"",
                   (20, s.imgh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

       #cv2.imwrite('dbg.png', img)
       s.writer.write(img)
       print('.',end='')

    def image(s, caption):
       img = np.zeros((s.imgh, s.imgw, 3), np.uint8)
       pic = s.pimg
       ph,pw = pic.shape[0:2]
       # scale it up
       fact = s.imgh/ph
       pic2 = cv2.resize(pic, (int(fact*pw),int(fact*ph)))
       ph,pw = pic2.shape[0:2]
       offx = (s.imgw-pw)//2
       img[0:ph, offx:offx+pw] = pic2

       cv2.putText(img, caption, (20,s.imgh-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   RED, 2)

       cv2.imwrite('dbg.png', img)
       print('.', end='')

       s.writer.write(img)


    def house(s, image_in=None, fat=None, extend=0, scale=1.0,
              drawVPs=False, drawVT=False, drawPB=False, drawPP=False,
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

       if drawVT or drawPB:
          tpX = scale_ip(img, s.vps['X'].ctr, s=scale); vpX=npt(tpX)
          tpY = scale_ip(img, s.vps['Y'].ctr, s=scale); vpY=npt(tpY)
          tpZ = scale_ip(img, s.vps['Z'].ctr, s=scale); vpZ=npt(tpZ)
          dXY = (vpY - vpX)*drawVT
          dYZ = (vpZ - vpY)*drawVT
          dZX = (vpX - vpZ)*drawVT
          wid=1 if drawPB else fat
          cv2.line(img, pt(img, vpX), pt(img, vpX+dXY), PPL, wid)
          cv2.line(img, pt(img, vpY), pt(img, vpY+dYZ), PPL, wid)
          cv2.line(img, pt(img, vpZ), pt(img, vpZ+dZX), PPL, wid)

          if drawPB:
             # we already know the principal point that all the
             # perpendicular bisectors pass through
             pp = (s.ppx,s.ppy)
             vX = (npt(line_inter(tpX,pp, tpY,tpZ)) - vpX)* drawPB
             vY = (npt(line_inter(tpY,pp, tpX,tpZ)) - vpY)* drawPB
             vZ = (npt(line_inter(tpZ,pp, tpX,tpY)) - vpZ)* drawPB
             pX = vpX+vX
             pY = vpY+vY
             pZ = vpZ+vZ
             wid=fat if fat is not None else 1
             cv2.line(img, pt(img,vpX), pt(img,pX), PPL, wid)
             cv2.line(img, pt(img,vpY), pt(img,pY), PPL, wid)
             cv2.line(img, pt(img,vpZ), pt(img,pZ), PPL, wid)

             if drawPB==1.0:
                vX *= -10.0 / np.linalg.norm(vX) # back up 10 pix
                vY *= -10.0 / np.linalg.norm(vY)
                vZ *= -10.0 / np.linalg.norm(vZ)
                Xv = npt((vX[1,0], -vX[0,0])) # perpendicular direction
                Yv = npt((vY[1,0], -vY[0,0]))
                Zv = npt((vZ[1,0], -vZ[0,0]))
                cv2.line(img, pt(img, pX + vX + Xv), pt(img, pX + vX), PPL, fat)
                cv2.line(img, pt(img, pY + vY + Yv), pt(img, pY + vY), PPL, fat)
                cv2.line(img, pt(img, pZ + vZ + Zv), pt(img, pZ + vZ), PPL, fat)
                cv2.line(img, pt(img, pX + vX + Xv), pt(img, pX + Xv), PPL, fat)
                cv2.line(img, pt(img, pY + vY + Yv), pt(img, pY + Yv), PPL, fat)
                cv2.line(img, pt(img, pZ + vZ + Zv), pt(img, pZ + Zv), PPL, fat)

       if drawPP:
          pp = ImgCircle((s.ppx,s.ppy), 10, PPL)
          pp.draw(img,s.cam)

       if drawVPs:
          for vp in s.vps.values():
             vp.draw(img, s.cam, w, extend, scale)

       #mark_origin(img, s.cam)
       if dump:   cv2.imwrite('dbg.png', img)
       if show:   cv2.imshow('House', img)
       if write:  s.writer.write(img)

       print('.', end='')


    def fan(s, extend, flen=0.5):
       img = np.zeros((s.imgh, s.imgw, 3), np.uint8)
       w = s.imgw
       h = s.imgh
       siz = w//8
       siz2 = (siz*3)//2
       ImgPoly(((0,0),(w,0),(w,siz),(0,siz)), SKY).draw(img,s.cam)
       ImgPoly(((0,siz),(w,siz),(w,h),(0,h)), BRN).draw(img, s.cam)

       fp0 = (w-3*siz,siz//2)
       fp1 = (w-2*siz,3*siz)
       cv2.line(img, fp0, fp1, WHT, 4)

       p0 = npt(fp0)
       p1 = npt(fp1)

       ppt = (p0+p1)/2
       fv = p1-p0
       pv = npt((fv[1],-fv[0]))
       f = ppt + pv*flen
       fpt = pt(img,f,flipy=False)

       cv2.circle(img, fpt, 10, WHT, -1)

       v = p0 - f
       a0 = m.atan2(v[1],v[0]) + 2*m.pi
       deg0 = m.degrees(a0)
       v = p1 - f
       a1 = m.atan2(v[1],v[0])
       deg1 = m.degrees(a1)
       pa = (a0+a1)/2
       da = a1-a0
       ddeg = m.degrees(da)
       da *= 0.95 / 22
       ddeg = m.degrees(da)

       for i in range(-11,12): #
          a = pa + i*da
          ddeg = m.degrees(a)
          v = npt((np.cos(a),np.sin(a)))
          r1 = 0.05*w
          r2 = 0.75*w
          q1 = f + r1*v
          q2 = f + r2*v
          q3 = q2 + extend*(q1-q2)
          if extend==1.0 and i==0:
             cv2.line(img, pt(img,q2,flipy=False), pt(img,q3,flipy=False), PPL, 2)
             cv2.circle(img, pt(img,ppt,flipy=False), 10, PPL, -1)
             cv2.circle(img, pt(img,ppt,flipy=False), 11, BLK, 1)
          else:
             cv2.line(img, pt(img,q2,flipy=False), pt(img,q3,flipy=False), WHT, 1)



       roof = ImgPoly(((siz,h-2*siz), (siz2,h-siz-siz2), (siz*2,h-2*siz)), BRN)
       roof.draw(img,s.cam)
       roof.col = YLW
       roof.filled = False
       roof.draw(img, s.cam)

       house = ImgPoly(((siz, h - siz), (siz, h - 2 * siz), (2 * siz, h - 2 * siz), (2 * siz, h - siz)), BRN)
       house.draw(img, s.cam)
       house.col = RED
       house.filled = False
       house.draw(img, s.cam)

       cv2.line(img, (siz,h-siz),(siz,h-2*siz),GRN,2)
       cv2.line(img, (2*siz,h-siz),(2*siz,h-2*siz),GRN,2)

       s.writer.write(img)
       print('.', end='')


    # render stuff into the image, and then pretend the image is in ground space
    # and view it from a perspective, so we can visualize the Perspective Pyramid
    def pip(s, f2,a2,t2,w2,x2,y2,z2, ppht=0.0):
       img = np.zeros((s.imgh, s.imgw, 3), np.uint8)
       K2  = buildK(f2, s.imgw//2, s.imgh//2)
       Rt2 = buildRt(a2, t2, w2, x2, y2, z2)
       cam2 = K2 @ Rt2
       for p in s.prims:
          p.drawdraw(img, s.cam, cam2)

       vpx = copy.deepcopy(s.vps['X'])
       vpy = copy.deepcopy(s.vps['Y'])
       vpz = copy.deepcopy(s.vps['Z'])

       for vp in (vpx,vpy,vpz):
          vp.flipy(s.imgh)

       ImgLine(vpx.ctr,vpy.ctr,PPL,'V').drawdraw(img, s.cam, cam2)
       ImgLine(vpy.ctr,vpz.ctr,PPL,'V').drawdraw(img, s.cam, cam2)
       ImgLine(vpz.ctr,vpx.ctr,PPL,'V').drawdraw(img, s.cam, cam2)

       for vp in (vpx,vpy,vpz):
          vp.drawdraw(img, s.cam, cam2)

       pp = ImgCircle((s.ppx, s.ppy), 3, PPL)
       pp.drawdraw(img, s.cam, cam2)

       for vp in (vpx,vpy,vpz):
          vp.flipy(s.imgh)

       apex = np.array([[s.ppx], [s.imgh-s.ppy], [-ppht], [1]])

       Line(D2to3(vpx.ctr), apex, PPL, 'P').draw(img, cam2, wid=3)
       Line(D2to3(vpy.ctr), apex, PPL, 'P').draw(img, cam2, wid=3)
       Line(D2to3(vpz.ctr), apex, PPL, 'P').draw(img, cam2, wid=3)
       aa = pt(img, cam2 @ apex, flipy=False)
       ImgCircle(aa,5,PPL).draw(img,cam2)







       cv2.imwrite('dbg.png', img)
       s.writer.write(img)
       print('.', end='')



    def __del__(s):
        s.writer.release()

    def __enter__(s):
        return s

    def __exit__(s, exc_type, exc_value, exc_traceback):
        s.writer.release()
        return True



#hw = HouseWriter(width=720, height=480)
hw = HouseWriter()
hw.updateRt()




print('\nArt',end='')
for a in range(10):
   hw.piranesi(picture=0.0, extend=1.0)   # just VP/VL
for f in np.arange(0.0, 1.01, 0.05):
   hw.piranesi(picture=f, extend=1.0) # fade in the picture
for e in np.arange(1.0, -0.01, -0.05):
   hw.piranesi(extend=e)              # shrink the vanishing lines
for w in (4,3,2,1):
   for a in range(5):
      hw.piranesi(extend=0.0, wid=w)  # thin the vanishing lines
for a in range(10):
   hw.piranesi(extend=0.0, wid=0)       # hold just the picture
for a in range(10):
   hw.piranesi(extend=0.0, wid=0, caption=True) # with URL

print('\nThis is a house',end='')
for a in range(30):
   hw.house()

print('\nXYZ',end='')
for fat_ax in ['X','Z','Y','D']:
   for a in range(15):
      hw.house(fat=fat_ax)

print('\nExtend',end='')
for e in np.arange(0.0, 1.01, 0.05):
   hw.house(extend=e)

print('\nShrink', end='')
for s in np.arange(1.0, 0.3, -0.05):
   hw.house(extend=1, scale=s, drawVPs=True)

print('\nTriangle', end='')
for e in np.arange(0.0, 1.01, 0.05):
   hw.house(extend=1, scale=0.3, drawVPs=True, drawVT=e, fat=4)

print('\nHeights to center', end='')
for e in np.arange(0.0, 1.0, 0.05):
   hw.house(extend=1, scale=0.3, drawVPs=True, drawVT=1.0, drawPB=e, fat=4)
for a in range(10):
   hw.house(extend=1, scale=0.3, drawVPs=True, drawVT=1.0, drawPB=1.0, fat=1)
for a in range(10):
   hw.house(extend=1, scale=0.3, drawVPs=True, drawVT=1.0, drawPB=1.0, fat=1, drawPP=True)
for a in range(10):
   hw.house(extend=1, scale=0.3, drawVPs=True, drawVT=1.0, drawPP=True)

print('\nPP Fan', end='')
for e in np.arange(0.0, 1.01, 0.02):
   hw.fan(e)
for a in range(30):
   hw.fan(1.0)


print('\nLooks like flat plane',end='')
x2 = -hw.imgw / 2
y2 = -hw.imgh / 2
for a in range(30):
   hw.pip(300,0,90,180, x2,y2,600, 500)

print('\nTilt to see pyramid',end='')
for a in range(25):
   hw.pip(300,a,90+a,180, x2,y2,600+3*a, 500)

# Hold this perspective
a2=25
t2=115
z2=675

print('\nNot too pointy',end='')
for a in range(30):
   hw.pip(300,a2,t2,180, x2,y2,z2, 500+30*a)
for a in range(30):
   hw.pip(300,a2,t2,180, x2,y2,z2, 500+30*(30-a))

print('\nNot too stubby',end='')
for a in range(15):
   hw.pip(300,a2,t2,180, x2,y2,z2, 500-30*a)
for a in range(15):
   hw.pip(300,a2,t2,180, x2,y2,z2, 500-30*(15-a))

print('\nApex has right angles',end='')
for a in range(30):
   hw.pip(300,a2,t2,180, x2,y2,z2, 500)

print('\nPaper', end='')
hw.pimg = cv2.imread('isprsfig.png')
for a in range(30):
   hw.image('Settergren, "Resection and Monte Carlo Covariance...", ISPRS 2020')
hw.pimg = cv2.imread('isprsmat.png')
for a in range(30):
   hw.image('Settergren, "Resection and Monte Carlo Covariance...", ISPRS 2020')



print('\nFocal length fan',end='')
for f in np.arange(0.75,0.25,-0.02):
   hw.fan(1.0, flen=f)

print('\nZoom out', end='')
for f in np.arange(300,100,-10):
   hw.updateK(f, None, None)
   hw.pip(300,a2,t2,180, x2,y2,z2, f)
for a in range(30):
   hw.pip(300,a2,t2,180, x2,y2,z2, 100)

print('\nZoom in', end='')
for f in np.arange(100,700,20):
   hw.updateK(f, None, None)
   hw.pip(300,a2,t2,180, x2,y2,z2, f)
for a in range(30):
   hw.pip(300,a2,t2,180, x2,y2,z2, 700)



hw.updateK(300, None, None)
print('\nZoom in picture', end='')
for f in np.arange(300, 700, 20):
   hw.updateK(f, None, None)
   hw.house()
for a in range(30):
   hw.house()

print('\nZoom out picture', end='')
for f in np.arange(700,99,-30):
   hw.updateK(f, None, None)
   hw.house()
for a in range(30):
   hw.house()




print('\nPush in', end='')
for z in np.arange(1.0, 0.59, -0.02):
   hw.updateRt(None, None, None, None, None, z)
   hw.house()
for a in range(30):
   hw.house()


print('\nStepped', end='')
steps=5
n=10 # frames per step (per foc/z)
f=lof=hw.foc; hif=700; df=(hif-lof)/steps/n
z=loz=hw.tz;  hiz=1.4; dz=(hiz-loz)/steps/n
for i in range(steps):
   for a in range(n):
      f += df
      hw.updateK(f, None, None)
      hw.house()
   if i == 0: # short pause after the first step
      for a in range(30):
         hw.house()
   for a in range(n):
      z += dz
      hw.updateRt(None, None, None, None, None, z)
      hw.house()
   if i == 0:
      for a in range(30):
         hw.house()



print('\nHitchcock',end='')
n = 100
lof = 300
loz = 0.8
for f,z in zip(np.arange(hif,lof,(lof-hif)/n),
               np.arange(hiz,loz,(loz-hiz)/n)):
   hw.updateK(f,None,None)
   hw.updateRt(None, None, None, None, None, z)
   hw.house()

print('\nVertigo', end='')
hw.pimg = cv2.imread('vertigo.jpg')
for a in range(30):
   hw.image('')

print('\nDog', end='')
hw.pimg = cv2.imread('dognose.png')
for a in range(30):
   hw.image('https://www.instagram.com/p/BSTPLebD7cp')


hw.writer.release()
