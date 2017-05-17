///************************************************
// *  Using
// *   (1) Move Detection
// *   (2) Open & Closed mouth
// *   (3) Optical Flow
// * to do face liveness detection
// *
// * Created date: 2017-03-28
// ************************************************/
//

#include "LandmarkCoreIncludes.h"
#include "LivenessDetection.hpp"

#include <fstream>
#include <sstream>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

#define THRESHOLD_ORIENTATION 0.6   //[0, 1], 面部中心到左右眼角的距离比， 越小越严
#define THRESHOLD_MOUSE 2.5         //[0, 无穷]， 嘴部面积的变化长宽比，越大越严
#define THRESHOLD_MOVE 17           //[0, 无穷]， 20帧算法的输出总和，越小越严
#define THRESHOLD_EYE 0.7           //[0，1】, 眼睛特征点变化的方差的比例， 越大越严
#define ORIENTATION_FRAME_NUM 20    //转头搜集的帧数
#define WIN_WIDTH 640
#define WIN_HEIGHT 480

using namespace cv;
using namespace std;

static void printErrorAndAbort( const std::string & error ){
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

vector<string> get_arguments(int argc, char **argv){
    vector<string> arguments;
    for(int i = 0; i < argc; ++i){
        arguments.push_back(string(argv[i]));
    }
    return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

void drawFaceOutline(Mat &display_image, Mat mask, Mat logo);
bool correctFacePosition(std::vector<Point2f> landmark);
Mat extractFaceMatrix(Mat temp, Rect result);
bool isRealFace(std::vector<float> closedMouse, std::vector<float> openedMouse);
float calculateUorV(Mat flow, int m_left, int m_right, int n_left, int n_right, int dim);
float opticalDifference(Mat flow);
bool mouseIdentificate(Mat &temp, std::vector<Point2f> landmark, std::vector<float> &closedMouse, std::vector<float> &openedMouse, int &numofClosedMouses, int &numofOpenedMouses,  Mat &mouseIdenImg);
bool orientationIdentificate(Mat temp, std::vector<Point2f> landmark, int &oriNum, float &sumOriValue,
                             bool &left_face, bool &right_face);
bool moveIdentificate(Mat temp, std::vector<Point2f> landmark, Mat &gray, Mat &preGray, Mat &flow,
                      int &optFlowPicNumber, std::vector<float> &optFlowDiffArray, Mat &moveIdenImg);
bool eyeIdentificate(Mat temp, bool detection_success, bool clnf_eye_model, std::vector<Point2f> landmarks, std::vector<std::vector<double>> &direction, int frame_count, bool &first_iden, Mat &eyeIdenImg);
void openFace2dlib_landmarks(std::vector<Point2f> &dlib_landmarks, std::vector<double> openface_landmarks);
double arrayDeviance(std::vector<double> arr);
void relativeLandmarks(const vector<Point2f> inputLandmarks, vector<Point2f> &outputLandmarks, LandmarkDetector::CLNF &clnf_model);

int main (int argc, char **argv){
    vector<string> arguments = get_arguments(argc, argv);
    // By default try webcam 0
    int device = 0;
    LandmarkDetector::FaceModelParameters det_parameters;
    // The modules that are being used for tracking
    LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
    det_parameters.track_gaze = false;
    double visualisation_boundary = 0.2;
    
//  判断模块开关: true 表示模块验证已经通过
    bool MoveIdentification = true;
    bool OrientationIdentification = true;
    bool MouseIdentification = true;
    bool EyeIdentification = false;
    //方向
    int oriNum = 0;
    float sumOriValue = 0.0;
    bool left_face = false;
    bool right_face = false;
    //嘴部
    std::vector<float> closedMouse = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};
    std::vector<float> openedMouse = {INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN};
    int numofClosedMouses = 0;
    int numofOpenedMouses = 0;
    //光流
    Mat gray, preGray, flow;
    int optFlowPicNumber = 0;
    std::vector<float> optFlowDiffArray(20,0.0);
    
    //眨眼
    std::vector<cv::Point3f> blink_eyes0;
    std::vector<cv::Point3f> blink_eyes1;
    bool first_iden = true;
    //存储脸部信息的标准差， 左右眼x,y,z各3个(3 * 2 = 6)，脸部周围轮廓18个
    std::vector<std::vector<double> > direction(36, std::vector<double>(19));
    
    //保存到相册
    bool writedImage = false;
    bool moveWriteImage = MoveIdentification;
    bool mouseWriteImage = MouseIdentification;
    bool eyeWriteImage = EyeIdentification;
    Mat moveIdenImg;
    Mat mouseIdenImg;
    Mat eyeIdenImg;
    
    // Do some grabbing
    cv::VideoCapture video_capture;
    
    INFO_STREAM( "Attempting to capture from device: " << device );
    video_capture = cv::VideoCapture( device );
    video_capture.set(CV_CAP_PROP_FRAME_WIDTH, WIN_WIDTH);
    video_capture.set(CV_CAP_PROP_FRAME_HEIGHT, WIN_HEIGHT);
    // Read a first frame often empty in camera
    
    if (!video_capture.isOpened()){
        FATAL_STREAM("Failed to open video source");
        return 1;
    }
    else INFO_STREAM( "Device or file opened");
    
    cv::Mat display_image;
    video_capture >> display_image;
    cv::flip(display_image, display_image, 1);
    int frame_count = 0;
    INFO_STREAM( "Starting tracking");
    double t_start = cvGetTickCount();
    Mat logo = imread("/Users/liuyan/Code/FaceLivenessDetection/FaceLivenessDetection/pic/faceOutline.jpg");
    Mat mask = imread("/Users/liuyan/Code/FaceLivenessDetection/FaceLivenessDetection/pic/faceOutline.jpg",0); //注意要是灰度图才行
    while(!display_image.empty())
    {
        cv::flip(display_image, display_image, 1);
        
        //if(frame_count%2 == 0) continue;
        double t_whole = cvGetTickCount();
        // Reading the images
        cv::Mat_<float> depth_image;
        cv::Mat_<uchar> grayscale_image;
        cv::Mat captured_image = display_image.clone();
        drawFaceOutline(display_image, mask, logo);
        
        if(captured_image.channels() == 3){
            cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
        }else{
            grayscale_image = captured_image.clone();
        }
        // The actual facial landmark detection / tracking
        bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);
        
        // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
        double detection_certainty = clnf_model.detection_certainty;
        // Only draw if the reliability is reasonable, the value is slightly ad-hoc
        if (detection_certainty < visualisation_boundary){
            LandmarkDetector::Draw(display_image, clnf_model);
        }
        
        std::vector<double> openface_landmarks = clnf_model.detected_landmarks;
        std::vector<cv::Point2f> landmarks;
        openFace2dlib_landmarks(landmarks, openface_landmarks);
        if(!clnf_model.detection_success){
            cv::putText(display_image, "Putting your face in the middle", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
        }else{
        
            //STEP 1.1: 光流
            if(!OrientationIdentification && !MouseIdentification && !MoveIdentification && !EyeIdentification){
                cv::putText(display_image, "STEP1 Identificating ... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                cv::putText(display_image, "Stare at your phone and don't move", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
                MoveIdentification = moveIdentificate(captured_image, landmarks, gray, preGray, flow, optFlowPicNumber, optFlowDiffArray, moveIdenImg);
            }
            //STEP 1.2: 转向
            if(!OrientationIdentification && !MouseIdentification && MoveIdentification && !EyeIdentification){
                cv::putText(display_image, "STEP1 Identificating ... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                OrientationIdentification = orientationIdentificate(display_image, landmarks, oriNum, sumOriValue, left_face, right_face);
            }
            
            //STEP2: 张闭嘴验证
            if(OrientationIdentification && !MouseIdentification && MoveIdentification && !EyeIdentification){
                cv::putText(display_image, "STEP2 Identificating ... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                cv::putText(display_image, "Move Your Mouse", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
                MouseIdentification = mouseIdentificate(captured_image, landmarks, closedMouse, openedMouse, numofClosedMouses, numofOpenedMouses,  mouseIdenImg);
            }
            
            //STEP4: 眨眼认证
            if (OrientationIdentification && MouseIdentification && MoveIdentification && !EyeIdentification){
                cv::Mat face = clnf_model.face_template;
                cv::putText(display_image, "STEP3 Identificating ... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                cv::putText(display_image, "Blink Eyes", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
                vector<cv::Point2f> relative_landmarks = landmarks;
                relativeLandmarks(landmarks, relative_landmarks, clnf_model);
                bool clnf_eye_model = clnf_model.eye_model;
                EyeIdentification = eyeIdentificate(captured_image, detection_success, clnf_eye_model, landmarks, direction, frame_count, first_iden, eyeIdenImg);
                //EyeIdentification = eyeIdentificate2(clnf_model, landmarks);
            }
            
        }
        //验证通过, 保存各步骤验证人脸, 输出结果
        if(OrientationIdentification && MouseIdentification && MoveIdentification && EyeIdentification){
            cv::putText(display_image, "Passed!", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
            if(!writedImage){
                double t_end = cvGetTickCount();
                cout<<"Time Consuming: "<<(t_end - t_start)/(double)cvGetTickFrequency()/1000000<<"s"<<endl;
                cout<<"Frame Used: "<<frame_count<<endl;
                std::cout<<"Detection Succeed, Wrting Photos"<<std::endl;
                if(!moveWriteImage)   cv::imwrite("movePic.jpg", moveIdenImg);
                if(!mouseWriteImage)  cv::imwrite("mousePic.jpg", mouseIdenImg);
                if(!eyeWriteImage) cv::imwrite("eyePic.jpg", eyeIdenImg);
                writedImage = true;
            }
        }
        
        if (!det_parameters.quiet_mode){
            cv::namedWindow("tracking_result");
            cv::imshow("tracking_result", display_image);
        }
        
        video_capture >> display_image;
        
        
        // detect key presses
        char character_press = cv::waitKey(1);
        
        // restart the tracker
        if(character_press == 'r'){
            clnf_model.Reset();
        }
        // quit the application
        else if(character_press=='q'){
            return(0);
        }
        // Update the frame count
        frame_count++;
        t_whole = ((double)cvGetTickCount() - t_whole) / cvGetTickFrequency() / 1000;
        cout<<"Frame Processing Time: "<<t_whole<<endl;
        
    }//end while(!captured_image.empty())
    frame_count = 0;
    // Reset the model, for the next video
    clnf_model.Reset();
    
    return 0;
}

void drawFaceOutline(Mat &display_image, Mat mask, Mat logo){
    cv::Mat display_image_clone = display_image.clone();
    float alpha = 0.5;
    float beta = 1-alpha;
    threshold(mask,mask,245,255,CV_THRESH_BINARY);
    cv::Mat imageROI;
    imageROI = display_image(cv::Rect(0,0,logo.cols,logo.rows));
    logo.copyTo(imageROI,mask);
    addWeighted( display_image, alpha, display_image_clone, beta, 0.0,display_image);
}

//提取脸部矩阵
Mat extractFaceMatrix(Mat temp, Rect result){
    Mat t_frame;
    float left_y = max(result.y, 0);
    float right_y = min(result.y+result.height, WIN_HEIGHT);
    float left_x = max(result.x, 0);
    float right_x = min(result.x+result.width, WIN_WIDTH);
    t_frame = temp(Range(left_y, right_y),Range(left_x, right_x));
    return t_frame;
}

//校正用户面部在摄像头部分的位置，校正的是左，右，底,中间方向的四个基准点
bool correctFacePosition(std::vector<Point2f> landmark){
    Point2f left_point = landmark[1];
    Point2f right_point = landmark[16];
    Point2f bottom_point = landmark[8];
    Point2f center_point = landmark[28];
    bool align_left = false;
    bool align_right = false;
    bool align_bottom = false;
    bool align_center = false;
    if( WIN_WIDTH/10 < left_point.x && left_point.x < WIN_WIDTH*2/5
       && WIN_HEIGHT/4 < left_point.y && left_point.y < WIN_HEIGHT/2 ) align_left = true;

    if( WIN_WIDTH*3/5 < right_point.x && right_point.x < WIN_WIDTH*9/10
       && WIN_HEIGHT/4 < right_point.y && right_point.y < WIN_HEIGHT/2 ) align_right = true;

    if( WIN_WIDTH/2-80 < bottom_point.x && bottom_point.x < WIN_WIDTH/2+80
       && WIN_HEIGHT*2/3 < bottom_point.y && bottom_point.y < WIN_HEIGHT*5/4 ) align_bottom = true;

    if( center_point.x >= WIN_WIDTH/4 && center_point.x <= WIN_WIDTH*3/4
       && center_point.y >= WIN_HEIGHT/4 && center_point.y <= WIN_HEIGHT*3/4) align_center = true;

    return align_left && align_right && align_bottom && align_center;
}

//检测用户是否按指令摇头
bool orientationIdentificate(Mat temp, std::vector<Point2f> landmark, int &oriNum,
                             float &sumOriValue, bool &left_face, bool &right_face){
    Point2f point_nose = landmark[33];
    Point2f point_left_eye = landmark[36];
    Point2f point_right_eye = landmark[45];
    
    
    float left_temp_x = (point_nose.x - point_left_eye.x) * (point_nose.x - point_left_eye.x);
    float left_temp_y = (point_nose.y - point_left_eye.y) * (point_nose.y - point_left_eye.y);
    
    float right_temp_x = (point_nose.x - point_right_eye.x) * (point_nose.x - point_right_eye.x);
    float right_temp_y = (point_nose.y - point_right_eye.y) * (point_nose.y - point_right_eye.y);
    
    float oriVal = (left_temp_x + left_temp_y) / (right_temp_x + right_temp_y);
    
    
    if(oriNum < ORIENTATION_FRAME_NUM){
        if(!left_face && !right_face){
            cv::putText(temp, "Turn Your Head to Right", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
        }else if(left_face && !right_face){
            cv::putText(temp, "Turn Your Head to Left", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
        }else{
            cv::putText(temp, "STEP1 Passed!!!", Point2f(WIN_WIDTH*3/4, WIN_HEIGHT/2), FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,0,255));
        }
        oriNum++;
        sumOriValue += oriVal;
    }else{
        
        double res = sumOriValue/oriNum;
        cout<<"Orientation output value "<<res<<endl;
        if(res > 1/THRESHOLD_ORIENTATION ){
            if(!left_face && !right_face) left_face = true;
        }
        if(res < THRESHOLD_ORIENTATION ){
            if(left_face && !right_face) right_face = true;
        }
        oriNum = 0;
        sumOriValue = 0.0;
    }
    return left_face && right_face;
}


//判断是否为真实人脸
bool isRealFace(std::vector<float> closedMouse, std::vector<float> openedMouse){
    auto biggestClosed = std::max_element(closedMouse.begin(), closedMouse.end());
    auto smallestOpened = std::min_element(openedMouse.begin(), openedMouse.end());
    cout<<"Biggest Mouse: "<<*biggestClosed<<endl;
    cout<<"Smallest Mouse: "<<*smallestOpened<<endl;
    cout<<"Value of Mouse Identification: "<<(*smallestOpened / *biggestClosed)<<endl;
    return (*smallestOpened / *biggestClosed) > THRESHOLD_MOUSE;
}

//通过嘴部的面积变化验证是否为真实人脸
bool mouseIdentificate(Mat &temp, std::vector<Point2f> landmark, std::vector<float> &closedMouse, std::vector<float> &openedMouse, int &numofClosedMouses, int &numofOpenedMouses, Mat &mouseIdenImg){
    bool isReal = false;
    std::vector<Point2f> mouseArea;
    for(int i = 48; i <= 60; i++){
        Point2f point = landmark[i];
        mouseArea.push_back(point);
    }
    
    int mouseSize = contourArea(mouseArea);
    float mouseLength = landmark[54].x - landmark[48].x;
    float mouseRatio = mouseSize/(mouseLength*mouseLength);
    
    //判断是否为真实人脸
    auto biggestClosed = std::max_element(closedMouse.begin(), closedMouse.end());
    auto smallestOpened = std::min_element(openedMouse.begin(), openedMouse.end());
    
    if(mouseRatio < *biggestClosed){
        int biggestIndex = (int)std::distance(closedMouse.begin(), biggestClosed);
        closedMouse[biggestIndex] = mouseRatio;
        numofClosedMouses++;
    }
    if(mouseRatio > *smallestOpened){
        int smallestIndex = (int)std::distance(openedMouse.begin(), smallestOpened);
        openedMouse[smallestIndex] = mouseRatio;
        numofOpenedMouses++;
    }
    if(numofClosedMouses >= 5 && numofOpenedMouses >=5){
        isReal = isRealFace(closedMouse, openedMouse);
    }
    mouseIdenImg = temp;
    return isReal;
}


// 计算光流
// 算法出自论文: <<A Liveness Detection Method for Face Recognition Based on Optical Flow Field>>
float opticalDifference(Mat flow){
    float optDifference = 0.0;
    int m = flow.rows;
    int n = flow.cols;

    //计算组件的值
    float U_left = calculateUorV(flow, 0, m/2, 0, n, 0);
    float V_left = calculateUorV(flow, 0, m/2, 0, n, 1);
    float U_right = calculateUorV(flow, m/2+1, m, 0, n, 0);
    float V_right = calculateUorV(flow, m/2+1, m, 0, n, 1);

    float U_upper = calculateUorV(flow, 0, m, 0, n/2, 0);
    float V_upper = calculateUorV(flow, 0, m, 0, n/2, 1);
    float U_lower = calculateUorV(flow, 0, m, n/2+1, n, 0);
    float V_lower = calculateUorV(flow, 0, m, n/2+1, n, 1);

    float U_center = calculateUorV(flow, 0, m, 0, n, 0);
    float V_center = calculateUorV(flow, 0, m, 0, n, 1);

    //计算系数的值
    float a1 = (U_right - U_left)/(m/2);
    float a2 = (V_right - V_left)/(m/2);

    float b1 = (U_upper - U_lower)/(n/2);
    float b2 = (V_upper - V_lower)/(n/2);

    float c1 = U_center - a1*n/2 - b1*m/2;
    float c2 = V_center - a2*n/2 - b2*m/2;

    //计算差值
    float numeratorofD = 0.0;
    float denominatorofD = 0.0;
    for(int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);
            float U_ij = flow_at_point[0];
            float V_ij = flow_at_point[1];
            float temp1 = (a1*i + b1*j + c1 - U_ij);
            float temp2 = (a2*i + b2*j + c2 - V_ij);
            numeratorofD += sqrt( temp1 * temp1 + temp2 * temp2 );
            denominatorofD += sqrt(U_ij* U_ij + V_ij*V_ij);
        }
    }
    optDifference = numeratorofD/denominatorofD;
    
    return optDifference;
}


/*Calculate horizontal or vertical component of the optical flow field
 * dim = 0 for horizontal, dim = 1 for vertical
 * m_left and m_right is the left boundary and right boundary of the cols and rows respectively
 */
float calculateUorV(Mat flow, int m_left, int m_right, int n_left, int n_right, int dim){
    float UorV = 0;
    for (int i=m_left; i<m_right; ++i) {
        for (int j=n_left; j<n_right; ++j) {
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);
            float f = flow_at_point[dim];
            UorV += f;
        }
    }
    UorV /= ((m_right-m_left)*(n_right-n_left));
    return UorV;
}

bool moveIdentificate(Mat temp, std::vector<Point2f> landmark, Mat &gray, Mat &preGray, Mat &flow, int &optFlowPicNumber, std::vector<float> &optFlowDiffArray,  Mat &moveIdenImg){
    //光流部分流程
    //计算光流部分的差值
    Point2f face_center = landmark[28];
    if ( (face_center.y - WIN_HEIGHT/4) > 0 && (face_center.y + WIN_HEIGHT/4) < WIN_HEIGHT &&
        (face_center.x - WIN_WIDTH/4) > 0  && (face_center.x + WIN_WIDTH/4) < WIN_WIDTH ){
        Mat faceArea = temp(Range(face_center.y - WIN_HEIGHT/6, face_center.y + WIN_HEIGHT/6),
                            Range(face_center.x - WIN_WIDTH/6, face_center.x + WIN_WIDTH/6) );
        cvtColor(faceArea, gray, CV_BGR2GRAY);
        if(preGray.data){
            calcOpticalFlowFarneback(preGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            float optDifference = opticalDifference(flow);
            optFlowDiffArray[optFlowPicNumber%20] = optDifference;
            //保存图像用于人脸识别
            if(optFlowPicNumber%10 == 0){
                moveIdenImg = temp;
            }
            optFlowPicNumber++;
        }
        std::swap(preGray, gray);
        if(optFlowPicNumber%20 == 0){
            float sumDif = 0;
            for(auto iter = optFlowDiffArray.begin(); iter != optFlowDiffArray.end(); iter++){
                sumDif += *iter;
            }
            
            cout<<"Value of Move Identification: "<<sumDif<<endl;
            if(sumDif < THRESHOLD_MOVE && sumDif > 0)   return true;
        }
    }
    return false;
}

//openface 特征点转dlib特征点
void openFace2dlib_landmarks(std::vector<Point2f> &dlib_landmarks, std::vector<double> openface_landmarks){
    for(int i = 0; i < 68; i++){
        Point2f dlib_i;
        float x = (float)openface_landmarks[i];
        float y = (float)openface_landmarks[68+i];
        dlib_landmarks.push_back(Point2f(x,y));
    }
}

//计算向量的方差
double arrayDeviance(std::vector<double> arr){
    double sum = 0.0;
    std::for_each (std::begin(arr), std::end(arr), [&](const double d) {
        sum  += d;
    });
    double mean = sum/arr.size();
    double accum  = 0.0;
    std::for_each (std::begin(arr), std::end(arr), [&](const double d) {
        accum  += (d-mean)*(d-mean);
    });
    double stdev = sqrt(accum/(arr.size()-1)); //方差
    return stdev;
}

//眨眼验证
bool eyeIdentificate(Mat temp, bool detection_success, bool clnf_eye_model, std::vector<Point2f> landmarks, std::vector<std::vector<double>> &direction, int frame_count, bool &first_iden, Mat &eyeIdenImg){
    if (detection_success){
        if(frame_count%20 == 0){
            std::vector<double> deviance_vec(36,0.0);
            for(int i = 0; i < 36; i++){
                deviance_vec[i] = arrayDeviance(direction[i]);
            }
            double deviance_eye = 0.0;
            double deviance_face = 0.0;
            for(int i = 1; i < 12; i++){ deviance_eye += deviance_vec[i];}
            for(int i = 12; i < 36; i++){ deviance_face += deviance_vec[i];}
            if(first_iden){
                first_iden = false;
                return false;
            }else{
                cout<<"Value of Eye Identification: "<<deviance_eye/deviance_face<<endl;
                return deviance_eye/deviance_face > THRESHOLD_EYE;
            }
            
        }else{
            for(int i = 0; i < 36; i++){
                if(i>=0 && i<12){
                    direction[i][frame_count%20 - 1] = landmarks[i+36].x + landmarks[i+36].y;
                }else{
                    direction[i][frame_count%20 - 1] = landmarks[i-12].x + landmarks[i-12].y;
                }
            }
        }
    }
    eyeIdenImg = temp;
    return false;
}

//眨眼验证
void relativeLandmarks(const vector<Point2f> inputLandmarks, vector<Point2f> &outputLandmarks, LandmarkDetector::CLNF &clnf_model){
    float relative_x = clnf_model.face_x_pos;
    float relative_y = clnf_model.face_y_pos;
    float relative_width = clnf_model.face_width;
    float relative_height = clnf_model.face_height;
    for( int i=0; i<inputLandmarks.size(); i++){
        outputLandmarks[i].x = (inputLandmarks[i].x - relative_x) / relative_width;
        outputLandmarks[i].y = (inputLandmarks[i].y - relative_y) / relative_height;
    }
}
