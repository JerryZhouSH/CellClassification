

OPENCV_INC_ROOT = /usr/local/include/opencv4 
OPENCV_LIB_ROOT = /usr/local/lib

INCLUDE_PATH = -I $(OPENCV_INC_ROOT)
LIB_PATH = -L $(OPENCV_LIB_ROOT)

OPENCV_LIB = -lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_imgcodecs

OBJS = smart_CellClassification.o
LIB = libsmart_cellclassification.so

CC = g++
CFLAGS  := -Wall -O3 -std=c++0x 

all : $(LIB)

%.o : %.cpp
	$(CC) $(CFLAGS) -fpic -c $< -o $@ $(INCLUDE_PATH)

$(LIB) : $(OBJS)
	rm -f $@
	g++ -shared -o $@ $(OBJS) $(LIB_PATH) $(OPENCV_LIB)
	rm -f $(OBJS)

tags :
	ctags -R *

clean:
	rm -f $(OBJS) $(TARGET) $(LIB)


