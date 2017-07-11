//
//  LivenessDetection.cpp
//  FaceLivenessDetection
//
//  Created by 刘岩 on 2017/5/12.
//  Copyright © 2017年 刘岩. All rights reserved.
//

#include "LivenessDetection.hpp"
namespace LD{
    void drawFaceOutline(Mat &temp){
        cv::Size sz(200,300);
        cv::Scalar color(255,255,255);
        cv::line(temp, Point2f(WIN_WIDTH/2, WIN_HEIGHT/2-40), Point2f(WIN_WIDTH/2, WIN_HEIGHT/2+50), color, 4);
        cv::line(temp, Point2f(WIN_WIDTH/2-55, WIN_HEIGHT/2-80),Point2f(WIN_WIDTH/2-115, WIN_HEIGHT/2-80), color, 4);
        cv::line(temp, Point2f(WIN_WIDTH/2+55, WIN_HEIGHT/2-80),Point2f(WIN_WIDTH/2+115, WIN_HEIGHT/2-80), color, 4);
        cv::line(temp, Point2f(WIN_WIDTH/2-40, WIN_HEIGHT/2+120),Point2f(WIN_WIDTH/2+40, WIN_HEIGHT/2+120), color, 4);
        cv::ellipse(temp, Point2f(WIN_WIDTH/2, WIN_HEIGHT/2), sz, 0, -20, 20, cv::Scalar(255,255,255), 4);
        cv::ellipse(temp, Point2f(WIN_WIDTH/2, WIN_HEIGHT/2), sz, 0, 160, 200, cv::Scalar(255,255,255), 4);
    }
}

