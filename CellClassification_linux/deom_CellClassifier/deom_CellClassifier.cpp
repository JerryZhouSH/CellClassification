/*****************************************************************************
* deom_CellClassifier.cpp: 此文件包含 "main" 函数。程序执行将在此处开始并结束。
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

#include <iostream>

#include <smart_CellClassification.h>

#define WIDTH 28
#define HEIGHT 28

#define MEAN_VALUE	127.5
#define SCALE_FACTOR (1.0/127.5)

int main(int argc, char* argv[])
{
	smart_CellClassification* classifier = new smart_CellClassification(WIDTH, HEIGHT, MEAN_VALUE, SCALE_FACTOR);

	int ret = classifier->createClassifier(cv::String(argv[1]));
	if (ret != 0) {
		std::cerr << "Create detection error..." << std::endl;
		return -1;
	}

	std::vector<cv::String> file_lists;
	std::cout << argv[1] << std::endl;
	cv::glob(cv::String(argv[2]) + "/*.png", file_lists);
	std::cout << "File Numbers : " << file_lists.size() << std::endl;

	for (int i = 0; i < file_lists.size(); i++) {
		cv::Mat input_img = cv::imread(cv::String(file_lists[i]));
		cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
		CELL_INFO_MAP info;
		ret = classifier->inference_img(input_img, info);
		if (ret != 0) {
			std::cerr << "Detecting objects return error..." << std::endl;
			return -1;
		}
		else {
			std::cout << "FileName: " << file_lists[i] << ", label: " << info.type << " , score: " << info.score << std::endl;
		}
	}

	delete classifier;

	return 0;
}
