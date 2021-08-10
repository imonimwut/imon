//
//  ViewController.swift
//  GazeTracking
//
//

import UIKit
import AVFoundation
import CoreML
import Vision
import ImageIO
import CoreGraphics
import CoreImage
import VideoToolbox

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private var captureSession: AVCaptureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    // CoreML and optical flow
    private var frame_count = 0
    private var prevImage: UIImage = UIImage()
    private let opticalFlowBridge = OpticalFlowBridge()
    private var latency_list: [Double] = []
    private let imageSize = CGSize(width: 112, height: 112)
    private var renderer = UIGraphicsImageRenderer()
    
    // Config
    private let toEnhance = false
    private let toTrackOpticalFlow = true
    private let imageEnhancementUNet = unet18_112().model
    private let gazeModel = gaze_mobilenetv2().model//gaze_efficientnetb3().model
    
    // Vision requests: Apple's face detection and facial landmark alignment
    private var detectionRequests: [VNDetectFaceRectanglesRequest]?
    private var trackingRequests: [VNTrackObjectRequest]?
    lazy var sequenceRequestHandler = VNSequenceRequestHandler()
    
    var captureDevice: AVCaptureDevice?
    var captureDeviceResolution: CGSize = CGSize()
    
    // Layer UI for drawing Vision results
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var roiImageView: UIImageView!
    var rootLayer: CALayer?
    var detectionOverlayLayer: CALayer?
    var detectedFaceRectangleShapeLayer: CAShapeLayer?
    var detectedFaceLandmarksShapeLayer: CAShapeLayer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        renderer = UIGraphicsImageRenderer(size: self.imageSize)
        // call camera session
        self.addCameraInput()
        self.getFrames()
        self.captureSession.sessionPreset = AVCaptureSession.Preset.vga640x480
        self.captureSession.startRunning()
        self.prepareVisionRequest()
    }
    
    private func addCameraInput() {
        guard let device = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInDualCamera, .builtInTrueDepthCamera],
            mediaType: .video,
            position: .front).devices.first else {
                fatalError("No front camera device found!!")
        }
        let cameraInput = try! AVCaptureDeviceInput(device: device)
        self.captureSession.addInput(cameraInput)
    }
    
    private func getFrames() {
        videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA)] as [String : Any]
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame.processing.queue"))
        self.captureSession.addOutput(videoDataOutput)
        guard let connection = self.videoDataOutput.connection(with: AVMediaType.video),
            connection.isVideoOrientationSupported else {return}
        connection.videoOrientation = .portrait
    }
    
    // MARK: Helper Methods for Handling Device Orientation & EXIF
    fileprivate func radiansForDegrees(_ degrees: CGFloat) -> CGFloat {
        return CGFloat(Double(degrees) * Double.pi / 180.0)
    }
    
    func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        switch deviceOrientation {
            case .portraitUpsideDown:
                return .rightMirrored
            case .landscapeLeft:
                return .downMirrored
            case .landscapeRight:
                return .upMirrored
            default:
                return .leftMirrored
        }
    }
    
    func exifOrientationForCurrentDeviceOrientation() -> CGImagePropertyOrientation {
        return exifOrientationForDeviceOrientation(UIDevice.current.orientation)
    }
    
    // MARK: Drawing Vision Observations
    func getRectFromPointArray(pointArray points:[CGPoint], parentImage image:CGImage) -> CGRect{
        var Xs:[Float] = [], Ys:[Float] = []
        for landmark in points{
            Xs.append(Float(landmark.x))
            Ys.append(Float(landmark.y))
        }
        let width = Xs.max()! - Xs.min()!
        let height = Ys.max()! - Ys.min()!
        let rect = CGRect(x: Int((Xs.min()!-0.25*width)*Float(image.width)),
                          y: Int((Ys.min()!-1.5*height)*Float(image.height)),
                                 width: Int(1.5*width*Float(image.width)),
                                 height: Int(4*height*Float(image.height)))
        return rect
    }
    
    fileprivate func setupVisionDrawingLayers() {
        let captureDeviceResolution = self.captureDeviceResolution
        let captureDeviceBounds = CGRect(x: 0, y: 0,
                                         width: captureDeviceResolution.width,
                                         height: captureDeviceResolution.height)
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX,
                                                     y: captureDeviceBounds.midY)
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)
        guard let rootLayer = self.rootLayer else {return}
        
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.masksToBounds = true
        overlayLayer.anchorPoint = normalizedCenterPoint
        overlayLayer.bounds = captureDeviceBounds
        overlayLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"
        faceRectangleShapeLayer.bounds = captureDeviceBounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint
        faceRectangleShapeLayer.fillColor = nil
        faceRectangleShapeLayer.strokeColor = UIColor.green.withAlphaComponent(0.7).cgColor
        faceRectangleShapeLayer.lineWidth = 5
        faceRectangleShapeLayer.shadowOpacity = 0.7
        faceRectangleShapeLayer.shadowRadius = 5
        
        let faceLandmarksShapeLayer = CAShapeLayer()
        faceLandmarksShapeLayer.name = "FaceLandmarksLayer"
        faceLandmarksShapeLayer.bounds = captureDeviceBounds
        faceLandmarksShapeLayer.anchorPoint = normalizedCenterPoint
        faceLandmarksShapeLayer.position = captureDeviceBoundsCenterPoint
        faceLandmarksShapeLayer.fillColor = nil
        faceLandmarksShapeLayer.strokeColor = UIColor.yellow.withAlphaComponent(0.7).cgColor
        faceLandmarksShapeLayer.lineWidth = 3
        faceLandmarksShapeLayer.shadowOpacity = 0.7
        faceLandmarksShapeLayer.shadowRadius = 5
        
        overlayLayer.addSublayer(faceRectangleShapeLayer)
        faceRectangleShapeLayer.addSublayer(faceLandmarksShapeLayer)
        rootLayer.addSublayer(overlayLayer)
        
        self.detectionOverlayLayer = overlayLayer
        self.detectedFaceRectangleShapeLayer = faceRectangleShapeLayer
        self.detectedFaceLandmarksShapeLayer = faceLandmarksShapeLayer
        
        self.updateLayerGeometry()
    }
    
    fileprivate func updateLayerGeometry() {
        guard let overlayLayer = self.detectionOverlayLayer,
            let rootLayer = self.rootLayer,
            let previewLayer = self.previewLayer
            else {return}
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let videoPreviewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        var rotation: CGFloat
        var scaleX: CGFloat
        var scaleY: CGFloat
        
        // Rotate the layer into screen orientation.
        switch UIDevice.current.orientation {
            case .portraitUpsideDown:
                rotation = 180
                scaleX = videoPreviewRect.width / captureDeviceResolution.width
                scaleY = videoPreviewRect.height / captureDeviceResolution.height
            case .landscapeLeft:
                rotation = 90
                scaleX = videoPreviewRect.height / captureDeviceResolution.width
                scaleY = scaleX
            case .landscapeRight:
                rotation = -90
                scaleX = videoPreviewRect.height / captureDeviceResolution.width
                scaleY = scaleX
            default:
                rotation = 0
                scaleX = videoPreviewRect.width / captureDeviceResolution.width
                scaleY = videoPreviewRect.height / captureDeviceResolution.height
        }
        
        // Scale and mirror the image to ensure upright presentation.
        let affineTransform = CGAffineTransform(rotationAngle: radiansForDegrees(rotation))
            .scaledBy(x: scaleX, y: -scaleY)
        overlayLayer.setAffineTransform(affineTransform)
        
        // Cover entire screen UI.
        let rootLayerBounds = rootLayer.bounds
        overlayLayer.position = CGPoint(x: rootLayerBounds.midX, y: rootLayerBounds.midY)
    }
    
    fileprivate func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D, to path: CGMutablePath, applying affineTransform: CGAffineTransform, closingWhenComplete closePath: Bool) {
        let pointCount = landmarkRegion.pointCount
        if pointCount > 1 {
            let points: [CGPoint] = landmarkRegion.normalizedPoints
            path.move(to: points[0], transform: affineTransform)
            path.addLines(between: points, transform: affineTransform)
            if closePath {
                path.addLine(to: points[0], transform: affineTransform)
                path.closeSubpath()
            }
        }
    }
    
    fileprivate func addIndicators(to faceRectanglePath: CGMutablePath, faceLandmarksPath: CGMutablePath, for faceObservation: VNFaceObservation) {
        let displaySize = self.captureDeviceResolution
        
        let faceBounds = VNImageRectForNormalizedRect(faceObservation.boundingBox, Int(displaySize.width), Int(displaySize.height))
        faceRectanglePath.addRect(faceBounds)
        
        if let landmarks = faceObservation.landmarks {
            // Landmarks are relative to -- and normalized within --- face bounds
            let affineTransform = CGAffineTransform(translationX: faceBounds.origin.x, y: faceBounds.origin.y)
                .scaledBy(x: faceBounds.size.width, y: faceBounds.size.height)
            
            // Treat eyebrows and lines as open-ended regions when drawing paths.
            let openLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEyebrow,
                landmarks.rightEyebrow,
                landmarks.faceContour,
                landmarks.noseCrest,
                landmarks.medianLine
            ]
            for openLandmarkRegion in openLandmarkRegions where openLandmarkRegion != nil {
                self.addPoints(in: openLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: false)
            }
            
            // Draw eyes, lips, and nose as closed regions.
            let closedLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEye,
                landmarks.rightEye,
                landmarks.outerLips,
                landmarks.innerLips,
                landmarks.nose
            ]
            for closedLandmarkRegion in closedLandmarkRegions where closedLandmarkRegion != nil {
                self.addPoints(in: closedLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: true)
            }
        }
    }
    
    /// - Tag: DrawPaths
    fileprivate func drawFaceObservations(_ faceObservations: [VNFaceObservation]) {
        guard let faceRectangleShapeLayer = self.detectedFaceRectangleShapeLayer,
            let faceLandmarksShapeLayer = self.detectedFaceLandmarksShapeLayer
            else {return}
        CATransaction.begin()
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectanglePath = CGMutablePath()
        let faceLandmarksPath = CGMutablePath()
        
        for faceObservation in faceObservations {
            self.addIndicators(to: faceRectanglePath,
                               faceLandmarksPath: faceLandmarksPath,
                               for: faceObservation)
        }
        
        faceRectangleShapeLayer.path = faceRectanglePath
        faceLandmarksShapeLayer.path = faceLandmarksPath
        
        self.updateLayerGeometry()
        
        CATransaction.commit()
    }
    
    // MARK: Performing Vision Requests
    fileprivate func prepareVisionRequest() {
        //self.trackingRequests = []
        var requests = [VNTrackObjectRequest]()
        let faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: { (request, error) in
            if error != nil {
                print("FaceDetection error: \(String(describing: error)).")
            }
            guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
                let results = faceDetectionRequest.results as? [VNFaceObservation] else {return}
            DispatchQueue.main.async {
                // Add the observations to the tracking list
                for observation in results {
                    let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                    requests.append(faceTrackingRequest)
                }
                self.trackingRequests = requests
            }
        })
        // Start with face detection
        self.detectionRequests = [faceDetectionRequest]
        self.sequenceRequestHandler = VNSequenceRequestHandler()
        self.setupVisionDrawingLayers()
    }
    
    // MARK: Process each frame
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags.readOnly)
        let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
        let width = CVPixelBufferGetWidth(imageBuffer) // 224 //
        let height = CVPixelBufferGetHeight(imageBuffer) // 224 //
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue
        bitmapInfo |= CGImageAlphaInfo.premultipliedFirst.rawValue & CGBitmapInfo.alphaInfoMask.rawValue
        let context = CGContext(data: baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo)
        guard let quartzImage = context?.makeImage() else {return}
        CVPixelBufferUnlockBaseAddress(imageBuffer, CVPixelBufferLockFlags.readOnly)
        let image = UIImage(cgImage: quartzImage)
        
        self.frame_count += 1
        NSLog("Frame \(self.frame_count)")
        // update imageView
        DispatchQueue.main.async {
            self.imageView.image = image.withHorizontallyFlippedOrientation()
        }
        
        let orientation = CGImagePropertyOrientation(rawValue: UInt32(UIDevice.current.orientation.rawValue))!
    
        guard let requests = self.trackingRequests, !requests.isEmpty else {
            // No tracking object detected, so perform initial detection
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: imageBuffer, orientation: orientation, options: [:])
            do {
                guard let detectRequests = self.detectionRequests else {
                    NSLog("Failed to create face detection request")
                    return
                }
                try imageRequestHandler.perform(detectRequests)
            } catch let error as NSError {
                NSLog("Failed to perform FaceRectangleRequest: %@", error)
            }
            return
        }
    
        do {
            try self.sequenceRequestHandler.perform(requests, on: imageBuffer, orientation: orientation)
        } catch let error as NSError {
            NSLog("Failed to perform SequenceRequest: %@", error)
        }
        
        // Facial landmark tracking request
        var newTrackingRequests = [VNTrackObjectRequest]()
        for trackingRequest in requests {
            guard let results = trackingRequest.results else {return}
            guard let observation = results[0] as? VNDetectedObjectObservation else {return}
            if !trackingRequest.isLastFrame {
                if observation.confidence > 0.3 {
                    trackingRequest.inputObservation = observation
                } else {
                    trackingRequest.isLastFrame = true
                }
                newTrackingRequests.append(trackingRequest)
            }
        }
        self.trackingRequests = newTrackingRequests
        
        // Nothing to track, so abort
        if newTrackingRequests.isEmpty {return}
        
        // Perform face landmark tracking on detected faces.
        var faceLandmarkRequests = [VNDetectFaceLandmarksRequest]()
        
        // Perform landmark detection on tracked faces.
        for trackingRequest in newTrackingRequests {
            let faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request, error) in
                if error != nil {
                    print("FaceLandmarks error: \(String(describing: error)).")
                }
                
                guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
                    let results = landmarksRequest.results as? [VNFaceObservation] else {return}
                // Perform all UI updates (drawing) on the main queue, not the background queue on which this handler is being called.
                DispatchQueue.main.async {
                    for faceObservation in results{
                        var faceRect = faceObservation.boundingBox
                        faceRect = CGRect(x: Int(faceRect.origin.x*CGFloat(quartzImage.width)), y: Int(faceRect.origin.y*CGFloat(quartzImage.height)),
                                              width: Int(faceRect.width*CGFloat(quartzImage.width)), height: Int(faceRect.height*CGFloat(quartzImage.height)))
//                        NSLog("Image size \(quartzImage.height) \(quartzImage.width) \(faceRect.height) \(faceRect.width)")
                        let faceImage = quartzImage.cropping(to: faceRect)!
                        
                        let leftEyeLandmarks = faceObservation.landmarks?.leftEye?.normalizedPoints
                        let rightEyeLandmarks = faceObservation.landmarks?.rightEye?.normalizedPoints
                        let leftEyeRect = self.getRectFromPointArray(pointArray: leftEyeLandmarks!, parentImage: faceImage)
                        let rightEyeRect = self.getRectFromPointArray(pointArray: rightEyeLandmarks!, parentImage: faceImage)
                        
                        // Crop eye images and resize
                        let leftEyeImage = faceImage.cropping(to: leftEyeRect)!
                        let rightEyeImage = faceImage.cropping(to: rightEyeRect)!
                        
                        var faceUIImage = UIImage(cgImage: faceImage)
                        faceUIImage = self.renderer.image { (context) in
                            faceUIImage.draw(in: CGRect(origin: .zero, size: CGSize(width: 224, height: 224)))
                        }
                        
                        var leftEyeUIImage = UIImage(cgImage: leftEyeImage)
                        var rightEyeUIImage = UIImage(cgImage: rightEyeImage)
                        leftEyeUIImage = self.renderer.image { (context) in
                            leftEyeUIImage.draw(in: CGRect(origin: .zero, size: self.imageSize))
                        }
                        rightEyeUIImage = self.renderer.image { (context) in
                            rightEyeUIImage.draw(in: CGRect(origin: .zero, size: self.imageSize))
                        }
                        
//                        self.roiImageView.image = leftEyeUIImage.withHorizontallyFlippedOrientation()
//                        NSLog(" left eye landmarks \(String(describing: leftEyeLandmarks))")
//                        NSLog(" right eye landmarks \(String(describing: rightEyeLandmarks))")
                        
                        // Optical flow tracking
                        if self.toTrackOpticalFlow{
                            if self.prevImage.size.height != 0{
                                let prevLandmarks = leftEyeLandmarks! + rightEyeLandmarks!
                                let trackedLandmarks = self.opticalFlowBridge.calcOpticalFlow(in: self.prevImage, current_image: faceUIImage)
                                
                            }
                            self.prevImage = faceUIImage// image (whole frame) or faceUIImage to apply tracking to the face region only
                        }
                        
                        // Image enhancement
                        if self.toEnhance{
                            do{
                                let leftEyeImageFeature = try MLFeatureValue(
                                    cgImage: leftEyeUIImage.cgImage!,
                                    constraint: self.imageEnhancementUNet.modelDescription.inputDescriptionsByName["input"]!.imageConstraint!,
                                    options: [.cropAndScale: VNImageCropAndScaleOption.centerCrop.rawValue]
                                )
                                let rightEyeImageFeature = try MLFeatureValue(
                                    cgImage: rightEyeUIImage.cgImage!,
                                    constraint: self.imageEnhancementUNet.modelDescription.inputDescriptionsByName["input"]!.imageConstraint!,
                                    options: [.cropAndScale: VNImageCropAndScaleOption.centerCrop.rawValue]
                                )
                                let leftEyeInput = try MLDictionaryFeatureProvider(
                                    dictionary: ["input": leftEyeImageFeature.imageBufferValue!]
                                )
                                let rightEyeInput = try MLDictionaryFeatureProvider(
                                    dictionary: ["input": rightEyeImageFeature.imageBufferValue!]
                                )
                                let input_array = [leftEyeInput, rightEyeInput]
                                let input_batch = MLArrayBatchProvider(array: input_array)
                                let batch_output = try self.imageEnhancementUNet.predictions(fromBatch: input_batch)
                                let enhancedLeftEyeImage = batch_output.features(at: 0).featureValue(for: "Identity")
                                let enhancedRighttEyeImage = batch_output.features(at: 1).featureValue(for: "Identity")
                            }
                            catch{
                                NSLog("Failed to perform eye-region image enhancement --!")
                            }
                        }
                        // Gaze estimation
                        do{
                            let mergedImageSize = CGSize(width: 224, height: 112)
                            UIGraphicsBeginImageContextWithOptions(mergedImageSize, false, 0.0)
                            leftEyeUIImage.draw(in: CGRect(x: 0, y: 0, width: 112, height: 112))
                            rightEyeUIImage.draw(in: CGRect(x: 112, y: 0, width: 224,  height: 112))
                            let mergedImage:UIImage = UIGraphicsGetImageFromCurrentImageContext()!
                            UIGraphicsEndImageContext()
    //                        self.roiImageView.image = mergedImage.withHorizontallyFlippedOrientation()
                            
                            let mergedImageFeature = try MLFeatureValue(
                                cgImage: mergedImage.cgImage!,
                                constraint: self.gazeModel.modelDescription.inputDescriptionsByName["input"]!.imageConstraint!,
                                options: [.cropAndScale: VNImageCropAndScaleOption.centerCrop.rawValue]
                            )
                            let gazeInput = try MLDictionaryFeatureProvider(
                                dictionary: ["input": mergedImageFeature.imageBufferValue!]
                            )
                            
                            let gazeOutput = try self.gazeModel.prediction(from: gazeInput)
//                            let gazeHeatmap = gazeOutput.featureValue(for: "Identity")
                        }
                        catch{
                            NSLog("Failed to perform gaze estimation --!")
                        }

                    }
                    self.drawFaceObservations(results)
                }
            })
            
            guard let trackingResults = trackingRequest.results else {return}
            
            guard let observation = trackingResults[0] as? VNDetectedObjectObservation else {return}
            let faceObservation = VNFaceObservation(boundingBox: observation.boundingBox)
            faceLandmarksRequest.inputFaceObservations = [faceObservation]
            
            // Continue to track detected facial landmarks.
            faceLandmarkRequests.append(faceLandmarksRequest)
            
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: imageBuffer, orientation: orientation, options: [:])
            
            do {
                try imageRequestHandler.perform(faceLandmarkRequests)
            } catch let error as NSError {
                NSLog("Failed to perform FaceLandmarkRequest: %@", error)
            }
        }
                
    }
}




