CC = c++
GCC = gcc
CFLAGS= -Os -DNO_CUDA -DNO_BILAT
#LIBSD = -L /c/lib/FLTK/lib -L /d/opencv-2.4.5/opencv-2.4.5/build-m64ati/install/lib
#LIBSD = -L/c/FLTK/lib -L/d/opencv-2.4.5/opencv-2.4.5/build-vanilla/install/lib
LIBS = -lfltk \
 -lfltk_forms \
 -lfltk_gl \
 -lfltk_images \
-lopencv_calib3d \
-lopencv_contrib \
-lopencv_core \
-lopencv_features2d \
-lopencv_flann \
-lopencv_gpu \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_legacy \
-lopencv_ml \
-lopencv_nonfree \
-lopencv_objdetect \
-lopencv_photo \
-lopencv_stitching \
-lopencv_superres \
-lopencv_ts \
-lopencv_video \
-lopencv_videostab 

all: cdb blindDeconv.app

cdb: caron16.o mainWin.o
	$(CC) -o cbd $(CFLAGS) caron16.o mainWin.o  `fltk-config --libs` /usr/local/lib/libopencv_contrib.a /usr/local/lib/libopencv_stitching.a /usr/local/lib/libopencv_nonfree.a /usr/local/lib/libopencv_superres.a /usr/local/lib/libopencv_ts.a /usr/local/lib/libopencv_videostab.a /usr/local/lib/libopencv_gpu.a /usr/local/lib/libopencv_legacy.a /usr/local/lib/libopencv_ml.a /usr/local/lib/libopencv_objdetect.a /usr/local/lib/libopencv_calib3d.a /usr/local/lib/libopencv_photo.a /usr/local/lib/libopencv_video.a /usr/local/lib/libopencv_features2d.a /usr/local/lib/libopencv_highgui.a /usr/local/share/OpenCV/3rdparty/lib/libIlmImf.a /usr/local/share/OpenCV/3rdparty/lib/liblibjasper.a /usr/local/share/OpenCV/3rdparty/lib/liblibtiff.a /usr/local/share/OpenCV/3rdparty/lib/liblibpng.a /usr/local/share/OpenCV/3rdparty/lib/liblibjpeg.a /usr/local/lib/libopencv_flann.a /usr/local/lib/libopencv_imgproc.a /usr/local/lib/libopencv_core.a /usr/local/share/OpenCV/3rdparty/lib/libzlib.a -framework AppKit -framework QuartzCore -framework QTKit -framework Cocoa 


caron16.o:  caron16.cpp caron16.h
	$(CC) -c $(CFLAGS)  caron16.cpp

mainWin.o:	mainWin.cpp
	$(CC) -c $(CFLAGS)  mainWin.cpp

blindDeconv.app:	cbd
	cp cbd BlindDeconv.app/Contents/MacOS/
	cp -r BlindDeconv.app /Applications

clean:	
	rm -f *.o cdb BlindDeconv.app/Contents/Resources/cbd


