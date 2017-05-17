//
//  LivenessDetection.hpp
//  FaceLivenessDetection
//
//  Created by 刘岩 on 2017/5/12.
//  Copyright © 2017年 刘岩. All rights reserved.
//

#ifndef LivenessDetection_hpp
#define LivenessDetection_hpp

#include <stdio.h>
#include "LandmarkCoreIncludes.h"

// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#define WIN_WIDTH 640
#define WIN_HEIGHT 480

using namespace std;
using namespace cv;

namespace LD{
    class LivenessDetection{
    public:
        void drawFaceOutline(cv::Mat &temp);
    };
}


#endif /* LivenessDetection_hpp */
