//
//  OpticalFlowBridge.h
//  GazeTracking
//
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface OpticalFlowBridge : NSObject

- (NSArray*) calcOpticalFlowIn: (UIImage *) prev_image current_image: (UIImage *) current_image;

@end
