/*****************************************************************************
* smart_CellClassification.cpp: Cell Classification library
*****************************************************************************
* Copyright (C) 2003-2008 CellDetection Project
*
* Authors: jerry_zhou@sina.com
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*****************************************************************************/

#include "smart_CellClassification.h"


smart_CellClassification::smart_CellClassification(int width, int height, float mean_val, float scale_factor)
	: mWidth(width),
	mHeight(height),
	mMeanVal(mean_val),
	mScaleFactor(scale_factor)
{
	std::cout << "Construct a new Cell Classifier ... " << std::endl;
}

smart_CellClassification::~smart_CellClassification()
{
	std::cout << "Destroy Cell Classifier ... " << std::endl;
}

int smart_CellClassification::createClassifier(cv::String model_path)
{
	if (model_path.c_str() == NULL) {
		std::cerr << "model is null..." << std::endl;
		return -1;
	}

	// Initialize network
	mNet = cv::dnn::readNetFromTensorflow(model_path);
	if (mNet.empty()) {
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "model: " << model_path << std::endl;
		return -1;
	}

	return 0;
}

int smart_CellClassification::inference_img(cv::Mat& img, CELL_INFO_MAP& cell_info)
{
	cv::Mat input_img;

	if (img.empty()) {
		std::cerr << "point of image or result is null..." << std::endl;
		return -1;
	}
	else {
		cv::Size dst_size = cv::Size(mWidth, mHeight);
		cv::resize(img, img, dst_size);
	}

	// Prepare blob
	img.convertTo(input_img, CV_32FC3);
	cv::Mat inputBlob = cv::dnn::blobFromImage(input_img, mScaleFactor, cv::Size(mWidth, mHeight),
		cv::Scalar(mMeanVal, mMeanVal, mMeanVal), false, false);

	// Set input blob
	mNet.setInput(inputBlob);

	// Make forward pass
	cv::Mat detection = mNet.forward();

	double confidence;
	cv::Point classIdPoint;

	cv::minMaxLoc(detection.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	cell_info.type = (CELL_LABEL_TYPE)classIdPoint.x;
	cell_info.score = (float)confidence;

	return 0;
}
