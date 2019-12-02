

OPENCV_INC_ROOT = /usr/local/include/opencv4 
OPENCV_LIB_ROOT = /usr/local/lib

LOCAL_LIB_ROOT = ../smart_CellClassification
LOCAL_INC_ROOT = ../smart_CellClassification

INCLUDE_PATH = -I $(OPENCV_INC_ROOT) -I $(LOCAL_INC_ROOT)
LIB_PATH = -L $(OPENCV_LIB_ROOT) -L $(LOCAL_LIB_ROOT)

OPENCV_LIB = -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
LOCAL_LIB = -lsmart_cellclassification

OBJS = deom_CellClassifier.o
TARGET = deom_CellClassifier

CC = g++
CFLAGS  := -Wall -O3 -std=c++0x 

%.o : %.cpp
	$(CC) $(CFLAGS) -fpic -c $< -o $@ $(INCLUDE_PATH)

$(TARGET) : $(OBJS)
	rm -f $@
	g++ -o $@ $(OBJS) $(LIB_PATH) $(OPENCV_LIB) $(LOCAL_LIB) 
	rm -f $(OBJS)

all : $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)


