//
//  OpticalFlow.hpp
//  GazeTracking
//
//

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class OpticalFlow {
    
public:
    /* Returns list of tracked points */
    vector<Point2f> calcOpticalFlowLK(Mat prev_image, Mat current_image, vector<Point2f> prev_pts);
    
    
};
