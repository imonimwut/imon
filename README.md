## iMon: Appearance-based Gaze Tracking System on Mobile Devices


### Datasets
GazeCapture: https://github.com/CSAILVision/GazeCapture

TabletGaze: http://sh.rice.edu/cognitive-engagement/tabletgaze/

To parse the meta-data including label data and the default face/eye position provided by GazeCapture dataset, please use function defined in mitdata_utils.py

### Eye-region image enhancement 
UNet model to enhance eye-region image is defined in image_enhancement.py

Flickr-Faces-HQ Dataset (FFHQ): https://github.com/NVlabs/ffhq-dataset

### Gaze estimation model
Gaze estimation models with iTracker and SAGE architectures and different CNN backbones (AlexNet, MobileNetV2, EfficientNetB3) are defined in gaze_model.py

Training and evaluation process are defined in gaze.py

