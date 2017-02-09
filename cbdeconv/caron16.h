#ifndef _DECONV_H_
#define _DECONV_H_
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#ifdef NO_CUDA
#ifdef HAVE_CUDA
#undef HAVE_CUDA
#endif
#endif
#ifdef NO_BILAT
#ifdef DO_BILAT
#undef DO_BILAT
#endif
#endif

using namespace cv;

class deconv {
	Mat PSF;
	Mat pv;
	Mat in;
	double nlmh;
	Mat result,prevM,resized;
#ifdef HAVE_CUDA
	gpu::GpuMat gprevM;
#endif
	double alpha,sigma;
	String fim,fout;
	int pType;
	vector<Mat> planes;
	 Mat img_yuv, comp;
	 bool force8;
	 double sizeR;
	
public:
	void getOrigrz(Mat& o);
	void setXY(Point xy);
	void preview();
	void getPreview(Mat& o);
	void deconvolution(const Mat& in);
	deconv(double alpha, double sigma);
	void getConv(Mat& out);
	void alphacallback(double a);
	void sigmacallback(double s);
	void bilatcallback(double b);
	void setPreview(const Mat& m);
	void filter(const Mat& m);
	void bilateralF(const Mat& m, Mat& o);
	int read(const char* inName,Size winSize);
	void runDeconvAndSave(const char* out);
	~deconv();
};
#endif

