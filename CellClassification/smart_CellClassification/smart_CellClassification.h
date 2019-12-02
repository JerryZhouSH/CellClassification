/*****************************************************************************
* smart_CellClassification.h: Cell Classification library
*****************************************************************************
* Copyright (C) 2003-2020 Cell Classification Project
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

#pragma once

#ifdef SMART_CELLCLASSIFICATION_EXPORTS
#define SMART_CELLCLASSIFICATION_EXPORTS __declspec(dllexport)
#else
#define SMART_CELLCLASSIFICATION_EXPORTS __declspec(dllimport)
#endif

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

typedef enum Cell_Label_Type {
	CELL_LABEL_TYPE_CELL = 0,
	CELL_LABEL_TYPE_NON_CELL = 1,
	CELL_LABEL_TYPE_MAX,
} CELL_LABEL_TYPE;

typedef struct Cell_Info_Map {
	CELL_LABEL_TYPE type;
	float score;
} CELL_INFO_MAP;

class SMART_CELLCLASSIFICATION_EXPORTS smart_CellClassification {
public:
	/**
	 * @brief Construct a new Cell Classifier object
	 *
	 * @param width 		image width
	 * @param height 		image height
	 * @param mean_val 		mean value that used to subtract the image data.
	 * @param scale_factor	scale factor for scaling the image data.
	 */
	explicit smart_CellClassification(int width = 28,
		int height = 28,
		float mean_val = 127.5,
		float scale_factor = 0.007843);
	/**
	 * @brief Destroy the Cell Classifier object
	 */
	~smart_CellClassification();

	/**
	 * @brief Initilaze a network with model.
	 *
	 * @param model     Path for tensorflow mode
	 * @return int      0 success, -1 failed.
	 */
	int createClassifier(cv::String model_path);

	/**
	 * @brief
	 *
	 * @param img 			The image that used to detect.
	 * @param result 		The integer array to saving result.
	 * @param save_result   If ture, draw the detect rangle on img.wz
	 * @return int 			0 success, -1 failed.
	 */
	int inference_img(cv::Mat& img, CELL_INFO_MAP& cell_info);

private:

	cv::dnn::Net mNet;

	int mWidth;
	int mHeight;

	float mMeanVal;
	float mScaleFactor;
};

