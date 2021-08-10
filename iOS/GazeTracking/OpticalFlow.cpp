//
//  OpticalFlow.cpp
//  GazeTracking
//
//

#include "OpticalFlow.hpp"

using namespace cv;
using namespace std;

//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/video.hpp>

vector<Point2f> OpticalFlow::calcOpticalFlowLK(Mat prev_image, Mat current_image, vector<Point2f> prev_pts){
    
    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    vector<Point2f> current_pts;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(prev_image, current_image, prev_pts, current_pts, status, err, Size(15,15), 2, criteria);
    
    return prev_pts;
}
