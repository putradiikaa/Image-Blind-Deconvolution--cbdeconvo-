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
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <fstream>
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Hor_Nice_Slider.H>
#include <FL/Fl_Shared_Image.H>
#include <string.h>
#include <errno.h>
#include <FL/Fl_File_Chooser.H>
#include <FL/fl_message.H>
#include <FL/Fl_Hor_Value_Slider.H>
#include <FL/fl_draw.h>
#include <FL/Fl_Native_File_Chooser.H>
#include "caron16.h"
using namespace cv;
#define WSIZE 600
#ifdef NO_BILAT
#ifdef DO_BILAT
#undef DO_BILAT
#endif
#else
#ifndef DO_BILAT
#define DO_BILAT
#endif
#endif
#ifndef FILTER_CPU
#define FILTER_CPU
#endif
deconv dd(0.35,1e-4);
string saveN = "";
string loadN = "";
string inputIMG = "";


Fl_Double_Window *w;
Fl_Button *loadB, *saveB, *quitB,*dcB;
Fl_Hor_Value_Slider *s1,*s2, *s3;
int bval = 100, wval = 140;
Size wins = Size(WSIZE,WSIZE);
char name[1024];


class double_blink_window : public Fl_Double_Window {
  void draw();
  int handle(int e);
public:
  bool ww;
  bool init;
  double_blink_window(int x, int y, int w,int h,const char *l, bool which)
    : Fl_Double_Window(x,y,w,h,l) {ww = which; resizable(this);}
};
double_blink_window *b1,*b2;
int double_blink_window::handle(int event) {
  if(event == FL_PUSH) {
    int x = Fl::event_x();
    int y = Fl::event_y();
    if(ww) {
      dd.setXY(Point(x,y));
      //std::cerr << "redraw" << std::endl;

      b2->redraw();


    }
    redraw();

  }
  return event;
}

void double_blink_window::draw(){
  //std::cerr << "draw " << ww << std::endl;
  Mat im3 = Mat::zeros(Size(WSIZE,WSIZE),CV_8UC3);
  Mat im1 = Mat::zeros(Size(WSIZE,WSIZE),CV_8UC1);
  Mat im;
  Mat oim;
  Mat t;
  if(ww) {
    dd.getOrigrz(oim);
    if(oim.channels()==3) 
      t= im3(Range(0,oim.rows),Range(0,oim.cols));
    else
      t= im1(Range(0,oim.rows),Range(0,oim.cols));
  }
  else {
    dd.getPreview(oim);
    if(oim.channels()==3) 
      t= im3(Range(0,oim.rows),Range(0,oim.cols));
    else
      t= im1(Range(0,oim.rows),Range(0,oim.cols));
  }

  oim.copyTo(t);
  if(oim.channels()==3)
    im = im3;
  else
    im = im1;

  fl_draw_image((const uchar*)im.data,0,0,im.cols,im.rows,im.channels());
  redraw();
}


void load_file(const char *n) {

  if (fl_filename_isdir(n)) {
    b1->label("@fileopen"); // show a generic folder
    b1->labelsize(64);
    b1->labelcolor(FL_LIGHT2);
    b1->image(0);
    b1->redraw();
    return;
  }

  //std::cerr << "load file " << n << std::endl;


  b1->init = false;
  b2->init = false;
  if(dd.read(n,wins)==0) {
    b1->label("@filenew"); // show an empty document
    b1->labelsize(64);
    b1->labelcolor(FL_LIGHT2);
    b1->image(0);
    b1->redraw();
    inputIMG = string(n);
    dd.setXY(Point(40,40));

    b1->redraw();
    b2->redraw();

    return;
  }
}
void save_file(const char *n) {

  if (fl_filename_isdir(n)) {
    b1->label("@fileopen"); // show a generic folder
    b1->labelsize(64);
    b1->labelcolor(FL_LIGHT2);
    b1->image(0);
    b1->redraw();
    return;
  }
  //std::cerr << "save file " << n << std::endl;
  dd.runDeconvAndSave(n);
}


void file_cb(const char *n) {
  if (!strcmp(name,n)) return;
  load_file(n);
  strcpy(name,n);
  w->label(name);
}

void fileS_cb(const char *n) {
  if (!strcmp(name,n)) return;
  save_file(n);
  strcpy(name,n);
  w->label(name);
}
void showErr(const char* errm) {
  std::cerr << errm << std::endl;
}

void lbutton_cb(Fl_Widget *,void *) {
  if(loadN != "") {
    dd.read(loadN.c_str(),wins);
    inputIMG = loadN;
    dd.setXY(Point(40,40));
    b1->redraw();
    b2->redraw();
    loadN = "";
    return;


  }


  Fl_Native_File_Chooser fload;
  fload.title("Open file");
  fload.type(Fl_Native_File_Chooser::BROWSE_FILE);
  fload.filter("Images\t*.{jpg,tif,png}");
  switch(fload.show()) {
  case -1: showErr(fload.errmsg()); break;
  case 1: return; break;
  default: loadN = String(fload.filename()); break;
  }
  b1->init = false;
  b2->init = false;
  if(dd.read(loadN.c_str(),wins)==0) {
    inputIMG = loadN;
    dd.setXY(Point(40,40));
    b1->redraw();
    b2->redraw();
    return;
  }
}





void sbutton_cb(Fl_Widget *,void *) {
  if(saveN != "") {
    dd.runDeconvAndSave(saveN.c_str());
    saveN = "";
    return;
  }
  Fl_Native_File_Chooser fsave;
  fsave.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE);  
  fsave.options(Fl_Native_File_Chooser::SAVEAS_CONFIRM);
  fsave.title("Save file");
  fsave.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE);
  fsave.filter("Images\t*.{jpg,tif,png}");
  switch(fsave.show()) {
  case -1: showErr(fsave.errmsg()); return; break;
  case 1: return; break;
  default: saveN = String(fsave.filename()); break;
  }
  dd.runDeconvAndSave(saveN.c_str());

}

void qbutton_cb(Fl_Widget *, void *) {
  exit(0);
}


void slider1_cb(Fl_Widget *, void *) {
  double a = s1->value();
  dd.alphacallback(a);
  b2->redraw();
}
void slider2_cb(Fl_Widget *, void *) {
  double s = s2->value();
  dd.sigmacallback(s);
  b2->redraw();
}
void slider3_cb(Fl_Widget *, void *) {
  double b = s3->value();
  dd.bilatcallback(b);
  b2->redraw();
}

int dvisual = 0;
int arg(int, char **argv, int &i) {
  if (argv[i][1] == '8') {dvisual = 1; i++; return 1;}
  return 0;
}

int main(int argc, char **argv) {
  int i = 1;
  int disable = 0;
  int is_osx_app = 0;
#ifdef __APPLE__
#define ARGS
#ifdef ARGS
  std::ofstream argum("/Users/mjkoskin/argum.txt");
  argum << "argc = " << argc << std::endl;
  for(int i=0; i< argc; i++)
    argum << i << " " << string(argv[i]) << std::endl;
  
#endif
  if(argc > 1) {
    if(strstr(argv[1],"-psn") != NULL) {
      if(argc == 3) {
	disable = 1;
	loadN = saveN = string(argv[3]);
      } else if(argc >= 4) {
	disable = 1;
	loadN = string(argv[4]);
	saveN = string(argv[5]);
      }
      is_osx_app = 1;
    }
      
  }
#ifdef ARGS
  argum << is_osx_app << std::endl;
  argum.close();
#endif

#endif
  
  if(argc == 2 && !is_osx_app) {
    disable = 1;
    loadN = saveN = string(argv[1]);
  }
  else if(argc == 3 && !is_osx_app) {
    disable = 1;
    loadN = string(argv[1]);
    saveN = string(argv[2]);
  }
  Fl::args(argc,argv,i,arg);


  Fl_Double_Window window(WSIZE*2+60,WSIZE+150);  ::w = &window;
  window.box(FL_THIN_DOWN_BOX);
  double_blink_window w2(10,10,WSIZE,WSIZE,"Main",true); ::b1=&w2;

  //w2.box(10,45,380,380);
  w2.box(FL_THIN_DOWN_BOX);
  w2.align(FL_ALIGN_INSIDE|FL_ALIGN_LEFT);

  //w2.box(WSIZE,45,380,380); ::b2 = &b2;

  w2.end();

  double_blink_window w3(WSIZE+20,10,WSIZE,WSIZE,"Deconv",false); ::b2=&w3;
  w3.box(FL_THIN_DOWN_BOX);
  w3.align(FL_ALIGN_INSIDE|FL_ALIGN_RIGHT);

  //  Fl_Button button(150,5,100,30,"load");
  // button.callback(button_cb);

  w3.end();
  Fl_Button load(10,WSIZE+70,50,20,"Load"); ::loadB = &load;
  Fl_Button save(60,WSIZE+70,50,20,"Save"); ::saveB = &save;
  Fl_Button quit(110,WSIZE+70,50,20,"Quit"); ::quitB = &quit;


  load.callback(lbutton_cb);
  save.callback(sbutton_cb);
  quit.callback(qbutton_cb);


  Fl_Hor_Value_Slider slid1(WSIZE+30,WSIZE+30,360,20,"alpha"); ::s1 = &slid1;
  slid1.callback(slider1_cb,NULL);

  s1->bounds(0.01,0.99);
  s1->value(0.3);
  Fl_Hor_Value_Slider slid2(WSIZE+30,WSIZE+30+40,360,20,"sigma"); ::s2 = &slid2;
  slid2.callback(slider2_cb,NULL);

  s2->bounds(0.00001,0.2);
  s2->value(0.001);
#ifdef DO_BILAT
  Fl_Hor_Value_Slider slid3(WSIZE+30,WSIZE+30+80,360,20,"bilat"); ::s3 = &slid3;
  slid3.callback(slider3_cb,NULL);
  s3->bounds(0.001,10.);
#endif
  window.end();
  Fl::visual(FL_RGB);
  window.show();
  w2.show();
  w3.show();

  if(loadN != "") 
    load_file(loadN.c_str());
  b1->redraw();
  return Fl::run();
}






