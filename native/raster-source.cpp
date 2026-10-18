#include <cmath>
#include <algorithm>
extern "C" void raster(const double* v, const int* f, int nf, const double* k, int w, int h, double* z){
 std::fill(z,z+w*h,1e30);
 for(int n=0;n<nf;n++){
  double x[3],y[3],zz[3]; bool ok=true;
  for(int j=0;j<3;j++){const double* p=v+3*f[n*3+j];zz[j]=p[2];if(p[2]<=1e-6){ok=false;break;}x[j]=k[0]*p[0]/p[2]+k[2];y[j]=k[4]*p[1]/p[2]+k[5];}
  if(!ok)continue;double den=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);if(std::abs(den)<1e-12)continue;
  int xmin=std::max(0,(int)std::floor(std::min({x[0],x[1],x[2]}))),xmax=std::min(w-1,(int)std::ceil(std::max({x[0],x[1],x[2]})));
  int ymin=std::max(0,(int)std::floor(std::min({y[0],y[1],y[2]}))),ymax=std::min(h-1,(int)std::ceil(std::max({y[0],y[1],y[2]})));
  for(int iy=ymin;iy<=ymax;iy++)for(int ix=xmin;ix<=xmax;ix++){
   double a=((y[1]-y[2])*(ix-x[2])+(x[2]-x[1])*(iy-y[2]))/den;
   double b=((y[2]-y[0])*(ix-x[2])+(x[0]-x[2])*(iy-y[2]))/den;double c=1-a-b;
   if(a>=-1e-9&&b>=-1e-9&&c>=-1e-9){double d=1/(a/zz[0]+b/zz[1]+c/zz[2]);int q=iy*w+ix;if(d<z[q])z[q]=d;}
  }
 }
}
