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
//#include "GazeEstimation.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"


#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

#define THRESHOLD_ORIENTATION 0.6
#define THRESHOLD_MOUSE 2.5
#define THRESHOLD_MOVE 17
#define THRESHOLD_EYE 0.2
#define THRESHOLD_FACE 200
#define ORIENTATION_FRAME_NUM 20
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


void drawFaceOutline(Mat &temp);
bool correctFacePosition(std::vector<Point2f> landmark);
//void landmarkExtraction(full_object_detection shape, std::vector<Point2f>& landmarks);
//Rect dlibRectangleToOpenCV(dlib::rectangle r);
//dlib::rectangle openCVRectToDlib(cv::Rect r);
Mat extractFaceMatrix(Mat temp, Rect result);
bool isRealFace(std::vector<float> closedMouse, std::vector<float> openedMouse);
float calculateUorV(Mat flow, int m_left, int m_right, int n_left, int n_right, int dim);
float opticalDifference(Mat flow);
bool mouseIdentificate(Mat &temp, std::vector<Point2f> landmark, std::vector<float> &closedMouse, std::vector<float> &openedMouse,
                       int &numofClosedMouses, int &numofOpenedMouses);
bool orientationIdentificate(Mat temp, std::vector<Point2f> landmark, int &oriNum, float &sumOriValue,
                             bool &front_face, bool &left_face, bool &right_face);
bool moveIdentificate(Mat temp, std::vector<Point2f> landmark, Mat &gray, Mat &preGray, Mat &flow,
                      int &optFlowPicNumber, std::vector<float> &optFlowDiffArray);
bool eyeIdentificate(bool detection_success, bool clnf_eye_model, std::vector<Point2f> landmarks, std::vector<std::vector<double>> &direction, int frame_count, bool &first_iden);
void openFace2dlib_landmarks(std::vector<Point2f> &dlib_landmarks, std::vector<double> openface_landmarks);
double arrayDeviance(std::vector<double> arr);

int main (int argc, char **argv){
    
    vector<string> arguments = get_arguments(argc, argv);
    
    // Some initial parameters that can be overriden from command line
    vector<string> files, depth_directories, output_video_files, out_dummy;
    
    // By default try webcam 0
    int device = 0;
    LandmarkDetector::FaceModelParameters det_parameters;
    
    // The modules that are being used for tracking
    LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
    
    // Grab camera parameters, if they are not defined (approximate values will be used)
    float fx = 0, fy = 0, cx = 0, cy = 0;
    // Get camera parameters
    LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);
    
    // If cx (optical axis centre) is undefined will use the image size/2 as an estimate
    bool cx_undefined = false;
    bool fx_undefined = false;
    if (cx == 0 || cy == 0){
        cx_undefined = true;
    }
    if (fx == 0 || fy == 0){
        fx_undefined = true;
    }
    
    // If multiple video files are tracked, use this to indicate if we are done
    bool done = false;
    int f_n = -1;
    
    det_parameters.track_gaze = true;
    
//    判断模块开关: true 表示模块验证已经通过
    bool OrientationIdentification = false;
    bool MouseIdentification = false;
    bool MoveIdentification = false;
    bool EyeIdentification = false;
    //方向
    int oriNum = 0;
    float sumOriValue = 0.0;
    bool front_face = false;
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
    while(!done){ // this is not a for loop as we might also be reading from a webcam
        string current_file;
        // We might specify multiple video files as arguments
        if(files.size() > 0){
            f_n++;
            current_file = files[f_n];
        }
        else{
            // If we want to write out from webcam
            f_n = 0;
        }
        
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
        
        cv::Mat captured_image;
        video_capture >> captured_image;
        
        // If optical centers are not defined just use center of image
        if (cx_undefined){
            cx = captured_image.cols / 2.0f;
            cy = captured_image.rows / 2.0f;
        }
        // Use a rough guess-timate of focal length
        if (fx_undefined){
            fx = 500 * (captured_image.cols / WIN_WIDTH);
            fy = 500 * (captured_image.rows / WIN_HEIGHT);
            fx = (fx + fy) / 2.0;
            fy = fx;
        }
        
        int frame_count = 0;
        
        INFO_STREAM( "Starting tracking");
        while(!captured_image.empty())
        {
            // Reading the images
            cv::Mat_<float> depth_image;
            cv::Mat_<uchar> grayscale_image;
            
            if(captured_image.channels() == 3){
                cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
            }
            else{
                grayscale_image = captured_image.clone();
            }
            // The actual facial landmark detection / tracking
            bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);
            
            // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
            double detection_certainty = clnf_model.detection_certainty;
            

            double visualisation_boundary = 0.2;
            
            // Only draw if the reliability is reasonable, the value is slightly ad-hoc
            if (detection_certainty < visualisation_boundary){
                LandmarkDetector::Draw(captured_image, clnf_model);
                
                double vis_certainty = detection_certainty;
                if (vis_certainty > 1)
                    vis_certainty = 1;
                if (vis_certainty < -1)
                    vis_certainty = -1;
            }
            
            // Work out the framerate
            if (frame_count % 10 == 0){
                double t1 = cv::getTickCount();
                fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
                t0 = t1;
            }
            
            std::vector<double> openface_landmarks = clnf_model.detected_landmarks;
            std::vector<cv::Point2f> landmarks;
            openFace2dlib_landmarks(landmarks, openface_landmarks);
            
            if(!correctFacePosition(landmarks)){
                cv::putText(captured_image, "Putting your face in the middle", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
            }else{
            
                //STEP1: 转向验证
            
                if(!OrientationIdentification && !MouseIdentification && !MoveIdentification && !EyeIdentification){
                    cv::putText(captured_image, "STEP1 Identificating... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                    OrientationIdentification = orientationIdentificate(captured_image, landmarks, oriNum, sumOriValue, front_face, left_face, right_face);
                }
            
                //STEP2: 张闭嘴验证
                if(OrientationIdentification && !MouseIdentification && !MoveIdentification && !EyeIdentification){
                    cv::putText(captured_image, "STEP2 Identificating... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                    MouseIdentification = mouseIdentificate(captured_image, landmarks, closedMouse, openedMouse, numofClosedMouses, numofOpenedMouses);
                }
                
                //STE3: 运动光流验证
                if(OrientationIdentification && MouseIdentification && !MoveIdentification && !EyeIdentification){
                    cv::putText(captured_image, "STEP3 Identificating (Don't Move)... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                    MoveIdentification = moveIdentificate(captured_image, landmarks, gray, preGray, flow, optFlowPicNumber, optFlowDiffArray);
                }
            
                //STEP4: 眨眼认证
                if (OrientationIdentification && MouseIdentification && MoveIdentification && !EyeIdentification){
                    cv::putText(captured_image, "STEP4 Identificating (Blink Eyes)... ", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                    bool clnf_eye_model = clnf_model.eye_model;
                    EyeIdentification = eyeIdentificate(detection_success, clnf_eye_model, landmarks, direction, frame_count, first_iden);
                    if(EyeIdentification){
                        cv::putText(captured_image, "STEP4 Passed", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                    }
                    else{
                        cv::putText(captured_image, "Blink Eyes", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
                    }
                }
                
            }
            //验证通过
            if(OrientationIdentification && MouseIdentification && MoveIdentification && EyeIdentification){
                cv::putText(captured_image, "Passed!", Point2f(10,20), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
            }
            
            
            // Write out the framerate on the image before displaying it
            char fpsC[255];
            std::sprintf(fpsC, "%d", (int)fps_tracker);
            string fpsSt("FPS:");
            fpsSt += fpsC;
            cv::putText(captured_image, fpsSt, cv::Point(560, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));
            
            if (!det_parameters.quiet_mode){
                cv::namedWindow("tracking_result", 1);
                cv::imshow("tracking_result", captured_image);
                
//                if (!depth_image.empty()){
//                    // Division needed for visualisation purposes
//                    imshow("depth", depth_image / 2000.0);
//                }
            }
            
            video_capture >> captured_image;
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
            
        }//end while(!captured_image.empty())
        
        frame_count = 0;
        
        // Reset the model, for the next video
        clnf_model.Reset();
        
        // break out of the loop if done with all the files (or using a webcam)
        if(f_n == files.size() -1 || files.empty()){
            done = true;
        }
        
    } //end while(done)
    
    return 0;
}


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


//// opencv Rect 转 dlib rectangle
//dlib::rectangle openCVRectToDlib(cv::Rect r){
//    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
//}
//
//// dlib rectangle 转 opencv Rect
//Rect dlibRectangleToOpenCV(dlib::rectangle r){
//    return cv::Rect(Point2i(r.left(), r.top()), Point2i(r.right() + 1, r.bottom() + 1) );
//}

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
bool orientationIdentificate(Mat temp, std::vector<Point2f> landmark, int &oriNum, float &sumOriValue,
                             bool &front_face, bool &left_face, bool &right_face){
    Point2f point_nose = landmark[33];
    Point2f point_left_eye = landmark[36];
    Point2f point_right_eye = landmark[45];


    float left_temp_x = (point_nose.x - point_left_eye.x) * (point_nose.x - point_left_eye.x);
    float left_temp_y = (point_nose.y - point_left_eye.y) * (point_nose.y - point_left_eye.y);

    float right_temp_x = (point_nose.x - point_right_eye.x) * (point_nose.x - point_right_eye.x);
    float right_temp_y = (point_nose.y - point_right_eye.y) * (point_nose.y - point_right_eye.y);

    float oriVal = (left_temp_x + left_temp_y) / (right_temp_x + right_temp_y);


    if(oriNum < ORIENTATION_FRAME_NUM){
        if(!front_face && !left_face && !right_face){
            cv::putText(temp, "FRONT", Point2f(WIN_WIDTH/3, WIN_HEIGHT/2), FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,0,255));
        }else if(front_face && !left_face && !right_face){
            cv::putText(temp, "LEFT", Point2f(WIN_WIDTH/3, WIN_HEIGHT/2), FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,0,255));
        }else if(front_face && left_face && !right_face){
            cv::putText(temp, "RIGHT", Point2f(WIN_WIDTH/3, WIN_HEIGHT/2), FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,0,255));
        }else{
            cv::putText(temp, "STEP1 Passed!!!", Point2f(WIN_WIDTH*3/4, WIN_HEIGHT/2), FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,0,255));
        }
        oriNum++;
        sumOriValue += oriVal;
    }else{
        if(sumOriValue/oriNum >= THRESHOLD_ORIENTATION && sumOriValue/oriNum <= 1/THRESHOLD_ORIENTATION){
            if(!front_face && !left_face && !right_face) front_face = true;
        }else if(sumOriValue/oriNum > THRESHOLD_ORIENTATION){
            if(front_face && !left_face && !right_face) left_face = true;
        }else{
            if(front_face && left_face && !right_face) right_face = true;
        }
        oriNum = 0;
        sumOriValue = 0.0;
    }
    return front_face && left_face && right_face;
}


//判断是否为真实人脸
bool isRealFace(std::vector<float> closedMouse, std::vector<float> openedMouse){
    auto biggestClosed = std::max_element(closedMouse.begin(), closedMouse.end());
    auto smallestOpened = std::min_element(openedMouse.begin(), openedMouse.end());
    cout<<"Biggest Mouse: "<<*biggestClosed<<endl;
    cout<<"Smallest Mouse: "<<*smallestOpened<<endl;
    return (*smallestOpened / *biggestClosed) > THRESHOLD_MOUSE;
}

//通过嘴部的面积变化验证是否为真实人脸
bool mouseIdentificate(Mat &temp, std::vector<Point2f> landmark, std::vector<float> &closedMouse, std::vector<float> &openedMouse,
                       int &numofClosedMouses, int &numofOpenedMouses){
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
        if(isReal == true){
            cv::putText(temp, "STEP1 Passed", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
        }
        else{
            cv::putText(temp, "Move Your Mouse", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
        }
    }else   cv::putText(temp, "Collecting Faces...", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0));
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

bool moveIdentificate(Mat temp, std::vector<Point2f> landmark, Mat &gray, Mat &preGray, Mat &flow, int &optFlowPicNumber, std::vector<float> &optFlowDiffArray){
    //光流部分流程
    //计算光流部分的差值
    Point2f face_center = landmark[28];
    //Mat faceArea = temp(Range(WIN_WIDTH/4, WIN_WIDTH*3/4),  Range(WIN_HEIGHT/4, WIN_HEIGHT*3/4));
    if ( (face_center.y - WIN_HEIGHT/4) >= 0 && (face_center.y + WIN_HEIGHT/4) <= WIN_HEIGHT &&
        (face_center.x - WIN_WIDTH/4) >= 0  && (face_center.x + WIN_WIDTH/4) <= WIN_WIDTH ){
        Mat faceArea = temp(Range(face_center.y - WIN_HEIGHT/4, face_center.y + WIN_HEIGHT*1/4),
                        Range(face_center.x - WIN_WIDTH/4, face_center.x + WIN_WIDTH*1/4));
        //Mat faceArea = temp(Range(169, 459), Range(79, 319));
        cvtColor(faceArea, gray, CV_BGR2GRAY);
        if(preGray.data){
            calcOpticalFlowFarneback(preGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            float optDifference = opticalDifference(flow);
            optFlowDiffArray[optFlowPicNumber%20] = optDifference;
            optFlowPicNumber++;
        }
        std::swap(preGray, gray);
        if(optFlowPicNumber%20 == 0){
            float sumDif = 0;
            for(auto iter = optFlowDiffArray.begin(); iter != optFlowDiffArray.end(); iter++){
                sumDif += *iter;
            }
            cout<<"sumDif"<<sumDif<<endl;
            if(sumDif < THRESHOLD_MOVE && sumDif > 0){
                cv::putText(temp, "STEP3 Passed", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
                return true;
            }
            else
                cv::putText(temp, "STEP3 NOT Passed", Point2f(10,40), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
        }
        //cout<<sumDif<<endl;
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
bool eyeIdentificate(bool detection_success, bool clnf_eye_model, std::vector<Point2f> landmarks, std::vector<std::vector<double>> &direction, int frame_count, bool &first_iden){
    
    //cout<<"frame_count"<<frame_count<<endl;
    if (detection_success /*&& clnf_eye_model*/){
        if(frame_count%20 == 0){
            std::vector<double> deviance_vec(36,0.0);
            for(int i = 0; i < 36; i++){
                deviance_vec[i] = arrayDeviance(direction[i]);
                //                cout<<"deviance "<<i<<": "<<deviance_vec[i]<<endl;
            }
            double deviance_eye = 0.0;
            double deviance_face = 0.0;
            for(int i = 1; i < 12; i++){ deviance_eye += deviance_vec[i];}
            for(int i = 12; i < 36; i++){ deviance_face += deviance_vec[i];}
            cout<<"deviance_eye: "<<deviance_eye<<endl;
            cout<<"deviance_face: "<<deviance_face<<endl;
            if(first_iden){
                first_iden = false;
                return false;
            }else{
                return deviance_eye/deviance_face > 0.7;
                //                return deviance_eye > THRESHOLD_EYE && deviance_face < THRESHOLD_FACE;
            }
            
        }else{
            for(int i = 0; i < 36; i++){
                if(i>=0 && i<12){
                    direction[i][frame_count%20 - 1] = landmarks[i+36].x + landmarks[i+36].y;
                }else{
                    direction[i][frame_count%20 - 1] = landmarks[i-12].x + landmarks[i-12].y;
                }
            }
            //cout<<gazeDirection0.x<<endl;
        }
    }
    return false;
}


