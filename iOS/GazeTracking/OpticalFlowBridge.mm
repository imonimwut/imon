//
//  OpticalFlowBridge.m
//  GazeTracking
//
//
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <Foundation/Foundation.h>
#import "OpticalFlowBridge.h"
#include "OpticalFlow.hpp"

@implementation OpticalFlowBridge

- (NSArray*) calcOpticalFlowIn: (UIImage *) prev_image current_image: (UIImage *) current_image {
    
    // convert uiimage to mat
    cv::Mat opencv_prev_image, opencv_current_image;
    UIImageToMat(prev_image, opencv_prev_image, true);
    UIImageToMat(current_image, opencv_current_image, true);
    // convert colorspace to the one expected by the optical flow algorithm RGB/GRAY
    cv::cvtColor(opencv_prev_image, opencv_prev_image, COLOR_BGR2GRAY);
    cv::cvtColor(opencv_current_image, opencv_current_image, COLOR_BGR2GRAY);
//
    // hardcoded example landmarks to track
    vector<cv::Point2f> prev_pts = vector<cv::Point2f>{{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0},
                                                        {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}};
//
//    // Run optical flow tracking
    OpticalFlow opticalFlow;
    vector<cv::Point2f> current_pts = opticalFlow.calcOpticalFlowLK(opencv_prev_image, opencv_current_image, prev_pts);
    
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:current_pts.size()];
    for (cv::Point2f tmpPoint:current_pts){
        NSValue *pointValue = [NSValue valueWithCGPoint:CGPointMake(tmpPoint.x, tmpPoint.y)];
        [result addObject:pointValue];
    }

    return result;
}

@end

