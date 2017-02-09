/*
  # Copyright (c) 2011 Matti Koskinen
  #
  # Version 1.3
  #
  # This program is free software; you can redistribute it and/or modify
  #  it under the terms of the GNU General Public License as published by
  # the Free Software Foundation; either version 2 of the License, or
  # (at your option) any later version.
  #
  # This program is distributed in the hope that it will be useful,
  # but WITHOUT ANY WARRANTY; without even the implied warranty of
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  # GNU General Public License for more details.
  #
  # You should have received a copy of the GNU General Public License
  # along with this program; if not, write to the Free Software
  # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 
  # USA
  #
  #
*/
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <Opencv2/Opencv.hpp>
#include <string.h>
#define WSIZE 600
#ifdef _MSC_VER

#include <Opencv2/gpu/gpu.hpp>
//#include <cuda.h>
//#include <cuda_runtime.h>
extern "C" {
#include "XGetopt.h"
}
#else
#include <libgen.h>
#endif
#include "caron16.h"
using namespace cv;
#define CUFFT
#define PLOT 0 //  0 no plots, 1 do plots
#define WIENER 0
#define FILTER_CPU
//#define IN_DENOISE
#ifdef NO_BILAT
#ifdef DO_BILAT
#undef DO_BILAT
#endif
#endif
#define FILTER_CPU
//#define BILAT_CPU
// structs for alpha and sigma searching
#if 0
class deconv {
public:
  void preview();
  void getPreview(Mat& o);
  void deconvolution(const Mat& in);
  deconv(double alpha, double sigma,String window);
  void getConv(Mat& out);
  void alphacallback(int pos);
  void sigmacallback(int pos);
  void bilatcallback(int pos);
  void setPreview(const Mat& m);
  void filter(const Mat& m);
  void bilateralF(const Mat& m, Mat& o);
  ~deconv();
private:
  Mat PSF;
  Mat pv;
  double nlmh;
  Mat result,prevM;
  gpu::GpuMat gprevM;
  double alpha,sigma;
  String window;
  int pType;
	
};
#endif
void deconv::setPreview(const Mat& m) {
  prevM = m.clone();
  Mat tmp;
  if(m.type()==CV_16UC1 || m.type()==CV_16UC3)
    pType = CV_16U;
  else
    pType = CV_8U;
	
  if(pType == CV_16U) 
    tmp  = m * 1./65536.;
  else
    tmp = m*1./256.;
#ifdef HAVE_CUDA
  gprevM.upload(tmp);
#else
  ;//cpuPrev = tmp;
#endif
	
}

deconv::deconv(double a, double s) {
  alpha = a;
  sigma = s;
  //window = w;
}
void deconv::getPreview(Mat& o) {
  o=pv.clone();
}

void deconv::filter(const Mat& in) {
  Mat in32;
  in.convertTo(in32,CV_32FC1);
  //	gpu::GpuMat filtin = gpu::GpuMat(in32);
  //	gpu::GpuMat filtout;

#ifdef FILTER_CPU
  Mat filtered;
  filter2D(in32,filtered,-1,PSF);
#else
  gpu::Stream st;
  st.waitForCompletion();
  gpu::filter2D(filtin, filtout,-1,PSF,Point(-1,-1),4,st);

  Mat filtered;
  filtered = Mat(filtout);
  filtout.release();
  filtin.release();
#endif
  //  Mat tresult = filtered.clone();
  filtered.convertTo(result,in.type());
  //std::cerr << "filtered" << std::endl;

}

void deconv::deconvolution(const Mat& in) {
  Mat fft32,fftcplx;
  in.convertTo(fft32,CV_32FC1);

  cv::dft(fft32,fftcplx,DFT_SCALE|DFT_COMPLEX_OUTPUT);
  //std::cerr << "cplx " << fftcplx.channels() << std::endl;
#ifdef HAVE_CUDA
  gpu::GpuMat ftmp = gpu::GpuMat(fftcplx);
  gpu::GpuMat mag,blur,power,powRes;
  
  gpu::magnitude(ftmp,mag);
  ftmp.release();
  //gpu::GaussianBlur(mag,blur,Size(5,5),1.4142);
  gpu::pow(mag,alpha,powRes);
  mag.release();
  blur.release();
  double maxv,minv;
  gpu::minMax(powRes,&minv,&maxv);
  Scalar s(maxv);
  
  gpu::divide(powRes,s,powRes);
  Mat outff = Mat(powRes);
  powRes.release();
  power.release();
#else
  Mat mag,blur,power,powRes;
  Mat cplanes[2];
  split(fftcplx,cplanes);
  magnitude(cplanes[0],cplanes[1],mag);
  pow(mag,alpha,powRes);
  double maxv,minv;
  minMaxLoc(powRes,&minv,&maxv);
  Scalar s(maxv);
  
  divide(powRes,s,powRes);
  Mat outff = powRes;
#endif
  Mat one = Mat::ones(Size(outff.size()),CV_32FC1);
  Mat pseudo(Size(outff.size()),CV_32FC1);
  
  Mat mask = abs(outff) > sigma;
  //std::cerr << "NZ " << countNonZero(mask) << std::endl;
  cv::divide(one,outff,pseudo);
  Mat pseudonz(Size(pseudo.size()),CV_32FC1);
  pseudo.copyTo(pseudonz,mask);
  Mat invfft(Size(pseudonz.size()),CV_32FC2),outfft;
  Mat planes[2];
  planes[0] = pseudonz;
  planes[1] = Mat::zeros(Size(pseudonz.size()),CV_32FC1);
  //  std::cerr << planes[0].size() << " " << planes[0].type() << " " << planes[1].size() << " " << planes[1].type() << std::endl;
  cv::merge(planes,2,invfft);
  int sq = in.rows;
  int sq1 = in.cols;
  cv::dft(invfft,outfft,DFT_INVERSE|DFT_REAL_OUTPUT|DFT_SCALE);
  Mat mpsf(Size(outfft.size()),CV_64FC1);
  Mat tmpsf = outfft.clone();
  cv::minMaxLoc(outfft,&minv,&maxv);
  if(maxv == 0.)
    outfft = Scalar::all(255);
  else
    outfft *= 65535.0/maxv;

  int ii = 0;
  int jj = 0;
  double max = 0.0;
  // demangle the output from  idft to get the psf in the center

  for(int i=0;i<sq1;i++) {
    for(int j=0;j<sq;j++) {
      int k,l;
      if(i >= sq1/2)
	k = i-(sq1/2);
      if(i<sq1/2)
	k = i+sq1/2;
      if(j>=sq/2)
	l = j-(sq/2);
      if(j<sq/2)
	l = j+sq/2;
      int idx1 = k+l*sq1;
      double tmp = tmpsf.at<float>(l,k); 
      mpsf.at<double>(jj,ii) = tmp;
      jj++;
    }
    jj = 0;
    ii++;
  }
  double mpmax,mpmin;
  minMaxLoc(mpsf,&mpmin,&mpmax);
  if(mpmax != 0) {
    Mat psfimg = mpsf.clone();
    psfimg *= 65535./mpmax;
    //imwrite("psf.png",psfimg);
  }
  int wpoint = in.cols /2 + 1;
  int hpoint = in.rows / 2 + 1;
#ifdef FILTER_CPU
  int psfSize = 91;
#else
  int psfSize = 15;
#endif
  mpsf /= sq1*sq;
  Mat spsf(psfSize,psfSize,CV_32FC1);
  // set the psf to the middle
  for(int i= -psfSize/2;i<psfSize/2+1;i++)
    for(int j=-psfSize/2;j<psfSize/2+1;j++) {
      double tmp = mpsf.at<double>(j+hpoint, i+wpoint);
      spsf.at<float>(j+psfSize/2, i+psfSize/2) = tmp;
    }
  //scale the inverse psf
  Scalar Sum = sum(spsf);
  double minf,maxf;
  minMaxLoc(spsf,&minf,&maxf);
  Mat clonepsf = spsf.clone();
  clonepsf *= 65535./maxf;
  //imwrite("spsf.png",clonepsf);
  spsf /=Sum(0);
  //std::cerr << spsf.size() << " " << Sum(0) << std::endl;
  /*  if(first) {
      std::ofstream plt("psf.txt");
      //plt.open();
      for(int pp=0;pp<spsf.rows;pp++) {
      for(int rr=0;rr<spsf.cols;rr++)
      plt << spsf.at<float>(pp,rr) << " ";
      plt << "\n";
      }
      plt.close();
      first = false;
      }
  */
  Mat itmp;
  //  gpu::Stream st;
  //  st.waitForCompletion();

  //  gpu::GpuMat filtout;
  //  gpu::GpuMat filtin;
  in.convertTo(itmp,CV_32FC1);
  //  filtin = gpu::GpuMat(itmp);
  //  gpu::GpuMat gpsf32;
  //Mat psf1 = mpsf(Range(10,mpsf.rows-10),Range(10,mpsf.cols-10));;
  Mat psf32;
  spsf.convertTo(psf32,CV_32FC1);
  PSF = psf32.clone();
  //  gpsf32 = gpu::GpuMat(psf32);
  //  //gpsf = gpu::GpuMat(spsf);
  //gpu::GpuMat filtin32,gpsf32;
  //  //filtin.convertTo(filtin32,CV_32F);
  //  //gpsf.convertTo(gpsf32,CV_32F);

  Mat filtered;
  filter2D(itmp,filtered,-1,PSF);
#if 0
  //  //  do the convolution between image and inverse psf in spatial space

  filtin = gpu::GpuMat(itmp);
  std::cerr << "filter2D" << std::endl;
  gpu::filter2D(filtin, filtout,-1, spsf,Point(-1,-1),4,st);
  std::cerr << "filter done" << std::endl;

  Mat filtered;
  filtered = Mat(filtout);
  //imshow("ff",filtered);
  filtout.release();
  filtin.release();
#endif

  Mat tresult = filtered.clone();
  tresult.convertTo(result,in.type());
  //std::cerr << "filtered" << std::endl;
}

void drawMarks(Mat& in) {
  putText(in,"OK",Point(30,30),FONT_HERSHEY_PLAIN,1.,CV_RGB(0,255,0),2);
  putText(in,"Reload",Point(60,30),FONT_HERSHEY_PLAIN,1.,CV_RGB(0,255,255),2);
  putText(in,"Cancel",Point(125,30),FONT_HERSHEY_PLAIN,1.,CV_RGB(255,0,0),2);
}


void deconv::preview() {
  if(prevM.empty())
    return;
  //Mat part;

  //if(prevM.rows > 600 && prevM.cols > 600) {
  //	part = prevM(Range(0,600),Range(0,600));
  // } else
  //part = prevM.clone();
  Mat mm = prevM.clone();
#ifdef DO_BILAT
  if(nlmh != 0.) {
#ifdef HAVE_CUDA
	  Mat t16;
	  if(pType == CV_16U)
		  t16 = prevM / 256;
    gpu::GpuMat tmp = gpu::GpuMat(t16);
    gpu::GpuMat outb;
    ;

    //gpu::FastNonLocalMeansDenoising nlm;
    //nlm.simpleMethod(tmp32, outb, nlmh);
    
    gpu::bilateralFilter(tmp,outb,5,nlmh,nlmh);
    mm = Mat(outb);
    if(pType == CV_16U)
      mm *= 256.;
    outb.release();
    //tmp32.release();
#else
    Mat tmp,outb;
    prevM.convertTo(tmp,CV_32F);
    if(pType==CV_16U)
      tmp *= 1./256;
    cv::bilateralFilter(tmp,outb,5,nlmh,nlmh);
    if(pType==CV_16U)
      mm = 256.*outb;
    else
      mm = outb;
#endif
  } else
	  mm = prevM.clone();
#endif
  deconvolution(mm);
  Mat pb(mm.size(),CV_32FC3);
  Mat n;
	
  Mat M[3];
  M[0]=result;
  M[1]=result;
  M[2]=result;
  merge(M,3,pb);
	
  double min,max;
  minMaxLoc(result,&min,&max);
  n = 255./max*pb;
  Mat b;
  n.convertTo(b,CV_8UC3);
  pv = b.clone();
}

void deconv::getConv(Mat& out) {
  out = result.clone();
}
void deconv::alphacallback(double a) {

	 
  if(a <= 0)
    alpha = 0.01;
  else
    alpha = a;
  //std::cerr << "pos " << pos << " alpha " << alpha << std::endl;
  preview();
}

void deconv::sigmacallback(double s) {
	
  if(s == 0)
    s = 1;

  sigma = s/1e1;
  preview();
}
void deconv::bilateralF(const Mat& in,Mat& out) {
#if defined(DO_BILAT) && defined(HAVE_CUDA)
  gpu::GpuMat tmp;
  Mat iTmp = in;
  if(pType == CV_16U) 
    iTmp *= 1./256.;
  tmp = gpu::GpuMat(iTmp);
  gpu::GpuMat outb,tmp32;
  tmp.convertTo(tmp32,CV_32F);
  Mat tmpout;
	
  gpu::bilateralFilter(tmp32,outb,10,nlmh,nlmh);
  tmpout = Mat(outb);
  if(pType == CV_16U)
    tmpout *= 256.;
  out = tmpout.clone();

  tmp.release();
  outb.release();
  tmp32.release();
#else
#ifndef DO_BILAT
  out = in;
#endif
  Mat tmp,outb;
  in.convertTo(tmp,CV_32F);
  if(pType == CV_16U)
    tmp * 1./256.;
  bilateralFilter(tmp,outb,5,nlmh,nlmh);
  if(pType==CV_16U)
    out = 256.*outb;
#endif
}

void deconv::bilatcallback(double b) {
  nlmh = b;
 #ifdef DO_BILAT	
  Mat n;
#ifdef HAVE_CUDA
  Mat t16;
  if(pType == CV_16U)
	  t16 = prevM/256;
  else
	t16 = prevM;
  gpu::GpuMat tmp = gpu::GpuMat(t16);
  gpu::GpuMat outb,tmp32;
  tmp.convertTo(tmp32,CV_8U);
  Mat tmpout;
  //gpu::FastNonLocalMeansDenoising nlm;
  //nlm.simpleMethod(tmp32, outb, nlmh);

  gpu::bilateralFilter(tmp32,outb,5,nlmh,nlmh);

  tmpout = Mat(outb);
  if(pType == CV_16U)
    tmpout *= 256.;
  prevM = tmpout;
  tmp.release();
  outb.release();
  tmp32.release();
  
#else
  Mat filtered;
  cv::bilateralFilter(prevM,filtered,5,nlmh,nlmh);
  prevM = filtered.clone();	
#endif
#endif
  preview();

}
deconv::~deconv() {
  ;
}


void usage() {
  fprintf(stderr,"usage: caron [flags] inputimage outputimage\n");
  fprintf(stderr,"flags:\n");
  fprintf(stderr,"\t -a alpha (0.65)\n");
  fprintf(stderr,"\t -B bilateral sigma (0.0)\n");
  fprintf(stderr,"\t -s sigma\n");
  fprintf(stderr,"\t -f filter size (int and odd)\n");
  fprintf(stderr,"\t -x x-position to read psf\n");
  fprintf(stderr,"\t -y y-pos to read psf\n");
  fprintf(stderr,"\t -n noise threshold\n");
  fprintf(stderr,"\t -H do histogram matching\n");
  fprintf(stderr,"\t -O force to 8-bit if inout 16-bit\n");
  fprintf(stderr,"\t -A search alpha and sigma automatically\n");
  fprintf(stderr,"\t -W use wiener or pseudoInverse\n");
}
#if 0
int main(int argc, char *argv[]) {
  // set default values
  double sigma = 1e-5;
  double alpha = 0.55;
  int filt = 3;
  int xpos = 0, ypos = 0;
  float  scaling = 0.;
  int ch;
  int ssize= 0;
  bool noiseThr = false;
  double position = 0.5;
  bool preview = false;
  bool histo = false;
  bool autom = false;
  bool force8 = false;
  bool wiener = false;
  double bilat = 0.;
  double nlmh = 5;
  if(argc < 2) {
    usage();
    std::cout << "usage " << std::endl;
    return -1;
  }
  // get parameters and values
  while((ch = getopt(argc,argv,"B:nws:a:p:f:x:y:q:PHNOS:AZ?")) != EOF) {
    switch(ch) {
    case 'A': autom = true;
      break;
    case 'B': nlmh = atof(optarg);
      break;
    case'w': wiener = true;
      break;
    case 'a': alpha = atof(optarg);
      break;
    case 'x': xpos = atoi(optarg);
      break;
    case 'y': ypos = atoi(optarg);
      break;
    case 'f': filt = atoi(optarg);
      break;
    case 's': sigma = atof(optarg);
      break;
    case 'P': preview = true;
      break;
    case 'O': force8 = true;
      break;
    case 'S': scaling = atof(optarg);
      break;
    case 'q': ssize = atoi(optarg);
      break;
    case 'p': position = atof(optarg);
      break;
    case 'n': noiseThr = true;
      break;
    case 'Z': ;
      break;
    case '?': usage(); return 0;
    default: usage();
      std::cout << "usage " << std::endl;
      return -1;
    }
  }
  cv::gpu::printShortCudaDeviceInfo(0);

  gpu::DeviceInfo dev_info(0);
  if (!dev_info.isCompatible())
    {
      std::cout << "GPU module isn't built for GPU #" << 0 << " ("
		<< dev_info.name() << ", CC " << dev_info.majorVersion()
		<< dev_info.minorVersion() << "\n";
      return -1;
    }
}
#endif
int  deconv::read(const char* inName, Size winSize) {
  // read the input image, if it fails, probably OpenCV can't handle input format
  fim = string(inName);
  in = imread(fim,-1);
 
  if(!in.data) {
    std::cout << "error reading input image" << std::endl;
    return -3;
  }
  if(in.cols < 600 || in.rows < 600)
   
	  
	  ; //return -4;



  Mat img = in;
  sizeR = 1.;
  if(in.rows > winSize.height || in.cols > winSize.width) {
    if(in.rows > in.cols) 
      sizeR = (double)winSize.height/in.rows;
    else
      sizeR = (double)winSize.width/in.cols;
    resize(in,resized,Size(0,0),sizeR,sizeR);
		
  } else
    resized = in;


   
  // if colour image split the colours to planes
  if(img.channels() == 3) {
    cvtColor(img,img_yuv,CV_BGR2YCrCb);
    split(img_yuv,planes);
    comp = planes[0];
  } else {
    // monochromatic
    comp = img;
  } 
  if(comp.type()==CV_16UC1 || comp.type()==CV_16UC3)
    pType = CV_16U;
  else
    pType = CV_8U;
  setXY(Point(comp.cols/2,comp.rows/2));
  //std::cerr << "read " << pType<< std::endl;
  return 0;
}
void deconv::getOrigrz(Mat& o) {
  Mat t,zz,c8;
  //std::cerr << "get orig " << pType << " " << CV_16U << " res " << resized.type() << std::endl;
  if(pType == CV_16U)
    t= resized / 256;
  else
    t = resized;
  t.convertTo(c8,CV_8U);
  if(c8.channels()==3) {
    Mat swp;
    Mat pln[3];
    split(c8,pln);
    Mat tmp = pln[0];
    pln[0]=pln[2];
    pln[2] = tmp;
    merge(pln,3,swp);
    zz = Mat::zeros(Size(WSIZE,WSIZE),CV_8UC3);
    tmp = zz(Range(0,t.rows),Range(0,t.cols));
    swp.copyTo(tmp);
    o = zz;
  } else {
    zz = Mat::zeros(Size(WSIZE,WSIZE),CV_8UC1);
    Mat tmp = zz(Range(0,t.rows),Range(0,t.cols));
    c8.copyTo(tmp);
    o = zz;
  }
}

void deconv::setXY(Point xy) {

  Mat prev;
  int cx,cy;
  xy.x = (int)(1./sizeR*xy.x);
  xy.y = (int)(1./sizeR*xy.y);
  //std::cerr << "xy " << xy.x << " " << xy.y << std::endl;


  if(in.cols > WSIZE && in.rows > WSIZE) {
    if(xy.x+WSIZE > in.cols)
      xy.x = in.cols-WSIZE-1;
    if(xy.y+WSIZE > in.rows)
      xy.y = in.rows-WSIZE-1;

    prev = comp(Range(xy.y,xy.y+WSIZE),Range(xy.x,xy.x+WSIZE));
  } else { 
    if(in.cols > WSIZE) 
      cx = WSIZE;
    else
      cx = in.cols;
    if(in.rows > WSIZE)
      cy = WSIZE;
    else
      cy = in.rows;
    prev = comp(Range(0,cy),Range(0,cx));
  }  
  setPreview(prev);
  alphacallback(alpha);
}

void deconv::runDeconvAndSave(const char* outName) {
  fout = string(outName);
 
 
  force8 = false;
  fout = string(outName);
  string lowercase = fout;
  std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), tolower);
  if(lowercase.find(".jpg") != string::npos) 
    force8 = true;
  
  if(nlmh != 0) {
    Mat bout;
    bilateralF(comp,bout);
    filter(bout);
  } else
    filter(comp);
  Mat out;
  getConv(out);
  Mat eight,out_yuv,mout,outZ;
  if(out.size() != in.size()) {
    resize(out,outZ,Size(in.size()));
    out = outZ;
  }
  // compose colour image back
  if(in.channels() == 3) {
    out.convertTo(planes[0],planes[1].type());
    merge(planes,img_yuv);
    cvtColor(img_yuv,out_yuv,CV_YCrCb2BGR);
    // if data 16-bit force to 8-bit if wanted
    if(force8 && out_yuv.depth() == CV_16U) {
      out_yuv *= 1./255.;
      out_yuv.convertTo(eight,CV_8U);
      cv::imwrite(fout,eight);
    } else 
      // write colour output
      cv::imwrite(fout,out_yuv);

  }
  // monchrome
  else {
    out.convertTo(mout,comp.type());
    if(force8 && mout.depth()==CV_16UC1) { //force to 8-bit
      eight = mout*1./255.;
      eight.convertTo(eight,CV_8U);
      cv::imwrite(fout,eight);
    } else {
      cv::imwrite(fout,out);
    }
  }
  std::cout << "success " << std::endl;

}
