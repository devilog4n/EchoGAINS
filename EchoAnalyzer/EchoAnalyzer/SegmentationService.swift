import UIKit
import CoreML
import Vision

class SegmentationService {

    // MARK: - Error Types
    enum SegmentationError: Error {
        case modelLoadingFailed(Error)
        case imagePreprocessingFailed(String)
        case visionRequestFailed(Error)
        case observationProcessingFailed(String)
        case outputTypeMismatch
    }

    // MARK: - Model and Input Properties
    private var coreMLModel: MLModel?
    private let inputWidth = 256 // As per plans.json
    private let inputHeight = 256 // As per plans.json
    // Name from the revised conversion script (onnx output name)
    private let outputFeatureName = "output_logits_onnx" 
    private let lvClassIndex = 1 // As per plans.json assumption
    
    // Normalization parameters from plans.json
    private let clipMin: Float32 = 0.0
    private let clipMax: Float32 = 222.0
    private let normMean: Float32 = 76.278633
    private let normStdDev: Float32 = 47.604141


    // MARK: - Initialization
    init() {
        loadModel()
    }

    private func loadModel() {
        do {
            // In a real app, EchoSegmenter.mlmodel (actually .mlmodelc directory)
            // would be compiled and added to the app bundle.
            // The VNCoreMLModel constructor handles finding the compiled model.
            // For this step, we just try to load the MLModel configuration.
            // The actual model file `EchoSegmenter.mlmodel` (or its compiled version)
            // is assumed to be in the app bundle.
            let modelConfig = MLModelConfiguration()
            // If using a specific model name from the bundle:
            // guard let modelURL = Bundle.main.url(forResource: "EchoSegmenter", withExtension: "mlmodelc") else {
            //     print("Error: EchoSegmenter.mlmodelc not found in bundle.")
            //     // Handle error appropriately - perhaps self.coreMLModel remains nil
            //     // and subsequent calls fail.
            //     return
            // }
            // self.coreMLModel = try MLModel(contentsOf: modelURL, configuration: modelConfig)
            
            // For now, we are using the generated class name for the model.
            // This assumes EchoSegmenter.mlmodel was added to the project and
            // Xcode generated the Swift class for it (e.g., EchoSegmenter).
            // If such a class isn't available because the model isn't physically present,
            // this approach won't work. We'll proceed as if it is.
            // If `EchoSegmenter()` (the generated class) is not found, it means the model
            // file wasn't actually added to the project and compiled.
             self.coreMLModel = try EchoSegmenter(configuration: modelConfig).model

            print("Core ML model 'EchoSegmenter' loaded successfully.")
        } catch {
            print("Error loading Core ML model 'EchoSegmenter': \(error)")
            self.coreMLModel = nil // Ensure model is nil if loading fails
            // Consider how to propagate this error - perhaps the service becomes non-functional.
        }
    }

    // MARK: - Segmentation
    func segment(frame: UIImage) async -> (UIImage?, MLMultiArray?, SegmentationError?) {
        guard let model = coreMLModel else {
            return (nil, nil, .modelLoadingFailed(NSError(domain: "SegmentationService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Core ML model is not loaded."])))
        }

        // 1. Preprocessing
        guard let cgImage = frame.cgImage else {
            return (nil, nil, .imagePreprocessingFailed("Could not get CGImage from input UIImage."))
        }

        // Resize, convert to grayscale, and create CVPixelBuffer
        // This needs to match the model's expected input format (e.g., 256x256 Grayscale Float32)
        // Normalization (e.g., /255.0) is specified in the Core ML model's input image type.
        guard let pixelBuffer = preprocessImage(cgImage: cgImage, width: inputWidth, height: inputHeight) else {
            return (nil, nil, .imagePreprocessingFailed("Failed to create CVPixelBuffer from UIImage."))
        }

        // 2. Inference
        do {
            let visionModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: visionModel) { (request, error) in
                // This completion handler is called after the request is performed.
                // The result processing will happen outside this async call.
            }
            request.imageCropAndScaleOption = .scaleFill // Match model's input aspect ratio handling

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try handler.perform([request]) // Synchronous perform on background thread from async func

            guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
                  let observation = observations.first else {
                return (nil, nil, .observationProcessingFailed("No observations or unexpected observation type returned."))
            }
            
            // Ensure the output feature name matches what the model provides.
            // The dummy conversion script used "output_mask_onnx".
            guard let multiArray = observation.featureValue.multiArrayValue else {
                 print("Debug: Observation feature value: \(observation.featureValue)")
                return (nil, nil, .outputTypeMismatch)
            }
            
            // 3. Output Handling (Initial)
            // For this subtask, return the raw MLMultiArray.
            // Optionally, try to create a basic mask image.
            // Assuming LV is class index 1 (Background=0, RV=1, Myo=2, LV=3 -> if LV = class 3, then index 3)
            // This needs to be confirmed from plans.json or model details.
            // For now, let's assume class index 1 for simplicity, or just return the raw array.
            // The output shape from the dummy model was (1, NumClasses, H, W)
            
            // 3. Process MLMultiArray output
            // Assuming LV class index is 1. This should be confirmed.
            let lvClassIndex = 1
            
            // 3. Process MLMultiArray output (logits)
            // Apply argmax to get a 2D array of class labels using the specified LV class index
            guard let classLabels = applyArgmax(to: multiArray, outputWidth: inputWidth, outputHeight: inputHeight) else {
                return (nil, multiArray, .observationProcessingFailed("Failed to apply argmax to model output (logits)."))
            }
            
            // Generate the LV-specific mask image from class labels
            let maskImage = generateLVMaskImage(classLabels: classLabels, 
                                                targetClassIndex: self.lvClassIndex, // Use defined LV class index
                                                width: inputWidth, 
                                                height: inputHeight)

            return (maskImage, multiArray, nil) // Still return raw multiArray for potential other uses

        } catch let requestError {
            return (nil, nil, .visionRequestFailed(requestError))
        }
    }

    // MARK: - Preprocessing Helpers
    private func preprocessImage(cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        // Create a CVPixelBuffer for Core ML input (Float32 Grayscale)
        // This function now also handles CLIPPING and Z-SCORE NORMALIZATION.

        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent32Float // Grayscale Float32
        ] as CFDictionary

        var cvPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent32Float, attributes, &cvPixelBuffer)
        guard status == kCVReturnSuccess, let unwrappedPixelBuffer = cvPixelBuffer else {
            print("Error: Failed to create CVPixelBuffer for preprocessing, status: \(status)")
            return nil
        }

        CVPixelBufferLockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(unwrappedPixelBuffer) else {
            print("Error: Could not get base address of CVPixelBuffer.")
            return nil
        }
        
        // 1. Draw original image into a temporary context to get raw pixel values (0-255)
        //    This ensures we are working with the image's pixel data in a known format.
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bytesPerRow = CVPixelBufferGetBytesPerRow(unwrappedPixelBuffer)
        
        var tempContextBuffer = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let tempContext = CGContext(
            data: &tempContextBuffer,
            width: width,
            height: height,
            bitsPerComponent: 8, // 8-bit grayscale for initial extraction
            bytesPerRow: bytesPerRow, // Should be `width` for 8-bit grayscale if tightly packed
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            print("Error: Failed to create temporary CGContext for raw pixel extraction.")
            return nil
        }
        tempContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // 2. Access the CVPixelBuffer's Float32 data pointer
        let buffer = baseAddress.assumingMemoryBound(to: Float32.self)

        // 3. Iterate, Clip, Normalize, and Fill CVPixelBuffer
        for y in 0..<height {
            for x in 0..<width {
                let tempBufferIndex = y * bytesPerRow + x // Index in 8-bit temp buffer
                var pixelValue = Float32(tempContextBuffer[tempBufferIndex]) // Raw pixel value (0-255)
                
                // Apply Intensity Clipping
                pixelValue = max(self.clipMin, min(pixelValue, self.clipMax))
                
                // Apply Z-Score Normalization
                pixelValue = (pixelValue - self.normMean) / self.normStdDev
                
                // Assign to Float32 CVPixelBuffer
                buffer[y * width + x] = pixelValue 
            }
        }
        return unwrappedPixelBuffer
    }
    
    // MARK: - Output Postprocessing Helper (Basic)
    
    // MARK: - Output Postprocessing Helpers

    /// Applies argmax to find the class with the highest score for each pixel.
    /// Assumes multiArray is shape [1, NumClasses, Height, Width] or [NumClasses, Height, Width]
    /// containing probabilities or logits.
    private func applyArgmax(to multiArray: MLMultiArray, outputWidth: Int, outputHeight: Int) -> [[UInt8]]? {
        // Validate shape
        guard multiArray.shape.count >= 3 else {
            print("Error: MLMultiArray shape \(multiArray.shape) not suitable for argmax.")
            return nil
        }

        // Determine dimensions, handling potential batch dimension
        let numClasses: Int
        let heightDimension: Int
        let widthDimension: Int

        if multiArray.shape.count == 4 && multiArray.shape[0].intValue == 1 { // Batch, C, H, W
            numClasses = multiArray.shape[1].intValue
            heightDimension = multiArray.shape[2].intValue
            widthDimension = multiArray.shape[3].intValue
        } else if multiArray.shape.count == 3 { // C, H, W
            numClasses = multiArray.shape[0].intValue
            heightDimension = multiArray.shape[1].intValue
            widthDimension = multiArray.shape[2].intValue
        } else {
            print("Error: Unexpected MLMultiArray shape \(multiArray.shape) for argmax.")
            return nil
        }
        
        guard widthDimension == outputWidth && heightDimension == outputHeight else {
            print("Error: MLMultiArray spatial dimensions (\(widthDimension)x\(heightDimension)) do not match expected (\(outputWidth)x\(outputHeight)).")
            return nil
        }

        var classLabels = [[UInt8]](repeating: [UInt8](repeating: 0, count: outputWidth), count: outputHeight)
        let ptr = UnsafeMutableBufferPointer<Float32>(start: multiArray.dataPointer.assumingMemoryBound(to: Float32.self), count: multiArray.count)

        for y in 0..<outputHeight {
            for x in 0..<outputWidth {
                var maxScore: Float32 = -Float.greatestFiniteMagnitude
                var maxClass: UInt8 = 0
                for c in 0..<numClasses {
                    // Indexing depends on the data layout: C * H * W + H * W + W
                    // Or more simply: (c * heightDimension * widthDimension) + (y * widthDimension) + x
                    let dataIndex = (c * heightDimension * widthDimension) + (y * widthDimension) + x
                    let score = ptr[dataIndex]
                    if score > maxScore {
                        maxScore = score
                        maxClass = UInt8(c)
                    }
                }
                classLabels[y][x] = maxClass
            }
        }
        return classLabels
    }

    /// Generates a UIImage mask where only the specified target class is colored.
    private func generateLVMaskImage(classLabels: [[UInt8]], targetClassIndex: Int, width: Int, height: Int) -> UIImage? {
        guard !classLabels.isEmpty && classLabels.count == height && classLabels[0].count == width else {
            print("Error: classLabels array dimensions do not match target width/height for mask generation.")
            return nil
        }

        var maskImageBuffer = [UInt8](repeating: 0, count: width * height * 4) // RGBA buffer

        for y in 0..<height {
            for x in 0..<width {
                let pixelClass = classLabels[y][x]
                let bufferIndex = (y * width + x) * 4

                if Int(pixelClass) == targetClassIndex { // Check against the targetClassIndex
                    maskImageBuffer[bufferIndex]     = 255  // Red (consistent with previous)
                    maskImageBuffer[bufferIndex + 1] = 0    // Green
                    maskImageBuffer[bufferIndex + 2] = 0    // Blue
                    maskImageBuffer[bufferIndex + 3] = 150  // Alpha (semi-transparent red for overlay)
                } else {
                    // Transparent for non-LV pixels
                    maskImageBuffer[bufferIndex]     = 0
                    maskImageBuffer[bufferIndex + 1] = 0
                    maskImageBuffer[bufferIndex + 2] = 0
                    maskImageBuffer[bufferIndex + 3] = 0
                }
            }
        }

        // Create UIImage from buffer
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let dataProvider = CGDataProvider(data: Data(maskImageBuffer) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: dataProvider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            print("Error: Failed to create CGImage from LV mask buffer.")
            return nil
        }
        return UIImage(cgImage: cgImage)
    }
}

// Placeholder for the Core ML model class generated by Xcode
// This would normally be auto-generated when you add EchoSegmenter.mlmodel to the project.
// If this file doesn't exist, the `SegmentationService.init()` will fail to load the model.
// e.g., class EchoSegmenter { init(configuration: MLModelConfiguration) throws { self.model = MLModel() } let model: MLModel }
// For now, I'll assume this class `EchoSegmenter` will be available via the (assumed) .mlmodel file.
// The `try EchoSegmenter(configuration: modelConfig).model` line will cause a compile error
// if `EchoSegmenter.mlmodel` is not in the project and its Swift class not generated.
```

**Important Note on `EchoSegmenter` class:**
The line `self.coreMLModel = try EchoSegmenter(configuration: modelConfig).model` in `SegmentationService.swift` assumes that an `EchoSegmenter.mlmodel` file has been added to the Xcode project, and Xcode has automatically generated a Swift class named `EchoSegmenter` for interacting with it.

If this model file and its generated class are not actually present (which they aren't, as I couldn't create the binary `.mlmodel`), this line would cause a compilation error ("Cannot find 'EchoSegmenter' in scope").

For the subtask to proceed as if the model exists, we are working under this assumption. The next steps will involve modifying `ContentView.swift` and `VideoDisplayView.swift` to use this service.

I will now modify `ContentView.swift`.
