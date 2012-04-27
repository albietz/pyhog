#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <string>
#include <vector>
#include <cmath>

using namespace boost::python;

#define PI 3.1415926535897931
double mind(double x, double y) { return (x <= y ? x : y); }

/* build lookup table a[] s.t. a[(dx+1.1)/2.2*(n-1)]~=acos(dx) */
float* acosTable() {
  int i, n=25000; float t, ni;
  static float a[25000]; static bool init=false;
  if( init ) return a; ni = 2.2f/(float) n;
  for( i=0; i<n; i++ ) {
    t = (i+1)*ni - 1.1f;
    t = t<-1 ? -1 : (t>1 ? 1 : t);
    a[i] = (float) acos( t );
  }
  init=true; return a;
}

struct MagOri
{
  MagOri(int msize, int osize) : mag(msize), ori(osize) {}

  std::vector<double> mag;
  std::vector<float> ori;
};

/* compute gradient magnitude and orientation at each location */
// h is actually w and vice versa
MagOri gradMagOri(std::string im, int h, int w, int d )
{
  MagOri mo(im.size(), im.size());
  std::vector<double>::iterator M = mo.mag.begin(), M0;
  std::vector<float>::iterator O = mo.ori.begin(), O0;

  int x, y, c, a=w*h;
  double m, m1, dx, dx1, dy, dy1, rx, ry; float o;
  std::string::iterator I = im.begin(), Ix, Ix0, Ix1, Iy0, Iy1;
  float *acost = acosTable(), acMult=(25000-1)/2.2f;

  int hd = h*d;
  for( x=0; x<w; x++ )
  {
    rx=.5; M0=M+x*h; O0=O+x*hd; Ix=I+x*hd; Ix0=Ix-hd; Ix1=Ix+hd;

    if(x==0) { Ix0=Ix; rx=1; }
    else if(x==w-1) { Ix1=Ix; rx=1; }

    for( y=0; y<h; y++ )
    {
      if(y==0) {   Iy0=Ix-0; Iy1=Ix+d; ry=1; }
      if(y==1) {   Iy0=Ix-d; Iy1=Ix+d; ry=.5; }
      if(y==h-1) { Iy0=Ix-d; Iy1=Ix+0; ry=1; }

      dy=(*Iy1-*Iy0)*ry; dx=(*Ix1-*Ix0)*rx;
      m=dx*dx+dy*dy;

      for(c=1; c<d; c++) {
        dy1=(*(++Iy1)-*(++Iy0))*ry;
        dx1=(*(++Ix1)-*(++Ix0))*rx;
        m1=dx1*dx1+dy1*dy1;
        if(m1>m) { m=m1; dx=dx1; dy=dy1; }
      }

      if( m==0 ) { o=0; }
      else {
        m=sqrt(m); /* o=acos(dx/m); */
        o = acost[(int)((dx/m+1.1f)*acMult)];
        if( o>PI-1e-5 ) o=0;
        else if( dy<0 ) o=(float)PI-o;
      }
      
      *(M0++) = m; *(O0++) = o;
      Ix0++; Ix1++; Iy0++; Iy1++; Ix++;
    }
  }
  return mo;
}

/* compute oBin gradient histograms per sBin x sBin block of pixels */
std::vector<double> gradHist( const std::vector<double>& mag, const std::vector<float>& ori, int h, int w, int d,
  int sBin, int oBin, bool sSoft, bool oSoft )
{
  std::vector<double>::const_iterator M = mag.begin();
  std::vector<float>::const_iterator O = ori.begin();

  const int hb=h/sBin, wb=w/sBin, h0=hb*sBin, w0=wb*sBin, nb=wb*hb;

  std::vector<double> hist(nb*oBin);
  std::vector<double>::iterator H = hist.begin();
  const double s=sBin, sInv=1/s, sInv2=1/s/s, oMult=(double)oBin/PI;
  std::vector<double>::iterator H0;
  int x, y, xy, o0, o1, xb0, yb0, oBin1=oBin*nb;
  double od0, od1, o, m, m0, m1, xb, yb, xd0, xd1, yd0, yd1;
  
  if( !sSoft || sBin==1 )
  {
    for( x=0; x<w0; x++ ) for( y=0; y<h0; y++ ) {
      /* interpolate w.r.t. orientation only, not spatial bin */
      xy=x*h+y;
      m=M[xy]*sInv2;
      o=O[xy]*oMult;
      o0=(int) o;

      m1=(o-o0)*m; m0=m-m1; o0*=nb; o1=o0+nb; if(o1==oBin1) o1=0;
      H0=H+(x/sBin)*hb+y/sBin; H0[o0]+=m0; H0[o1]+=m1;
    }
    return hist;
  }
  
  for( x=0; x<w0; x++ ) for( y=0; y<h0; y++ ) {
    /* get interpolation coefficients */
    xy=x*h+y; m=M[xy]*sInv2; o=O[xy]*oMult; o0=(int) o;
    xb=(((double) x)+.5)*sInv-0.5; xb0=(xb<0) ? -1 : (int) xb;
    yb=(((double) y)+.5)*sInv-0.5; yb0=(yb<0) ? -1 : (int) yb;
    xd0=xb-xb0; xd1=1.0-xd0; yd0=yb-yb0; yd1=1.0-yd0; H0=H+xb0*hb+yb0;
    /* interpolate using bilinear or trilinear interpolation */
    if( !oSoft || oBin==1 ) {
      o0*=nb;
      if( xb0>=0 && yb0>=0     ) *(H0+o0)      += xd1*yd1*m;
      if( xb0+1<wb && yb0>=0   ) *(H0+hb+o0)   += xd0*yd1*m;
      if( xb0>=0 && yb0+1<hb   ) *(H0+1+o0)    += xd1*yd0*m;
      if( xb0+1<wb && yb0+1<hb ) *(H0+hb+1+o0) += xd0*yd0*m;
    } else {
      od0=o-o0; od1=1.0-od0; o0*=nb; o1=o0+nb; if(o1==oBin1) o1=0;
      if( xb0>=0 && yb0>=0     ) *(H0+o0)      += od1*xd1*yd1*m;
      if( xb0+1<wb && yb0>=0   ) *(H0+hb+o0)   += od1*xd0*yd1*m;
      if( xb0>=0 && yb0+1<hb   ) *(H0+1+o0)    += od1*xd1*yd0*m;
      if( xb0+1<wb && yb0+1<hb ) *(H0+hb+1+o0) += od1*xd0*yd0*m;
      if( xb0>=0 && yb0>=0     ) *(H0+o1)      += od0*xd1*yd1*m;
      if( xb0+1<wb && yb0>=0   ) *(H0+hb+o1)   += od0*xd0*yd1*m;
      if( xb0>=0 && yb0+1<hb   ) *(H0+1+o1)    += od0*xd1*yd0*m;
      if( xb0+1<wb && yb0+1<hb ) *(H0+hb+1+o1) += od0*xd0*yd0*m;
    }
  }

  return hist;
}

/* compute HOG features given gradient histograms */
tuple hog( const std::vector<double>& hist, int h, int w, int d, int sBin, int oBin ) {
  std::vector<double>::iterator N, N1, HG1;
  double n;
  int o, x, y, x1, y1, hb, wb, nb, hb1, wb1, nb1;
  double eps = 1e-4/4.0/sBin/sBin/sBin/sBin; /* precise backward equality */
  hb=h/sBin; wb=w/sBin; nb=wb*hb; hb1=hb-2; wb1=wb-2; nb1=hb1*wb1;

  if(hb1<=0 || wb1<=0)
  {
    return make_tuple(std::vector<double>(), 0, 0, 0);
  }

  std::vector<double> norm(nb);
  std::vector<double> hog(nb1*oBin*4);
  std::vector<double>::iterator HG = hog.begin();
  std::vector<double>::const_iterator H = hist.begin(), H1;

  N = norm.begin();

  for(o=0; o<oBin; o++) for(x=0; x<nb; x++)
    N[x]+=H[x+o*nb]*H[x+o*nb];

  for( x=0; x<wb1; x++ ) for( y=0; y<hb1; y++ ) {
    HG1 = HG + x*hb1 + y; /* perform 4 normalizations per spatial block */
    
    for(x1=1; x1>=0; x1--) for(y1=1; y1>=0; y1--) {
      N1 = N + (x+x1)*hb + (y+y1);  H1 = H + (x+1)*hb + (y+1);
      n = 1.0/sqrt(*N1 + *(N1+1) + *(N1+hb) + *(N1+hb+1) + eps);
      for(o=0; o<oBin; o++) { *HG1=mind(*H1*n, 0.2); HG1+=nb1; H1+=nb; }
    }
  }
  return make_tuple(hog, hb1, wb1, oBin*4);
}

tuple pyhog(std::string im, int w, int h, int d, int sBin=8, int oBin=9)
{
  MagOri mo = gradMagOri(im, w, h, d);
  return hog(gradHist(mo.mag, mo.ori, w, h, d, sBin, oBin, true, true), w, h, d, sBin, oBin);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(pyhog_overloads, pyhog, 4, 6);

BOOST_PYTHON_MODULE(pyhog)
{	
	class_<std::vector<double> >("double_vector")
        .def(vector_indexing_suite<std::vector<double> >())
    ;

	def("hog", pyhog, pyhog_overloads());
}
