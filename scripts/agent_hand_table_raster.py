"""Perspective-correct Warp CPU rasterizer for bounded hand pose search."""
import warp as wp
@wp.kernel
def triangles(vertices:wp.array(dtype=wp.vec3),faces:wp.array2d(dtype=wp.int32),depth:wp.array(dtype=float),width:int,height:int,fx:float,fy:float,cx:float,cy:float):
 i=wp.tid();a=vertices[faces[i,0]];b=vertices[faces[i,1]];c=vertices[faces[i,2]]
 if a[2]>.05 and b[2]>.05 and c[2]>.05:
  ax=fx*a[0]/a[2]+cx;ay=fy*a[1]/a[2]+cy;bx=fx*b[0]/b[2]+cx;by=fy*b[1]/b[2]+cy;cx0=fx*c[0]/c[2]+cx;cy0=fy*c[1]/c[2]+cy
  xmin=wp.max(0,int(wp.floor(wp.min(ax,wp.min(bx,cx0)))));xmax=wp.min(width-1,int(wp.ceil(wp.max(ax,wp.max(bx,cx0)))));ymin=wp.max(0,int(wp.floor(wp.min(ay,wp.min(by,cy0)))));ymax=wp.min(height-1,int(wp.ceil(wp.max(ay,wp.max(by,cy0)))));den=(by-cy0)*(ax-cx0)+(cx0-bx)*(ay-cy0)
  if wp.abs(den)>1.e-7:
   for y in range(ymin,ymax+1):
    for x in range(xmin,xmax+1):
     u=((by-cy0)*(float(x)-cx0)+(cx0-bx)*(float(y)-cy0))/den;v=((cy0-ay)*(float(x)-cx0)+(ax-cx0)*(float(y)-cy0))/den;w=1.-u-v
     if u>=0. and v>=0. and w>=0.:
      inverse=u/a[2]+v/b[2]+w/c[2]
      if inverse>0.:wp.atomic_min(depth,y*width+x,1./inverse)
