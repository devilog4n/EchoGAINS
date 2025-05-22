import SwiftUI
import UIKit // For UIImage

struct ContentView: View {
    // MARK: - State Variables
    @State private var selectedVideoURL: URL?
    @State private var videoFileName: String = "No video selected"
    @State private var ejectionFractionString: String = "--%"
    
    @State private var extractedFrames: [UIImage] = []
    @State private var segmentationMasks: [UIImage?] = []

    // Frame Navigation
    @State private var currentFrameIndex: Int = 0
    @State private var displayedFrame: UIImage? = nil
    @State private var displayedMask: UIImage? = nil

    // UI State
    @State private var isProcessingVideo: Bool = false
    @State private var isSegmenting: Bool = false
    @State private var isCalculatingAFC: Bool = false
    
    // Error Handling
    @State private var processingError: String? = nil
    @State private var segmentationError: String? = nil
    @State private var afcError: String? = nil

    // AFC Results
    @State private var lvedAreaPixels: Int? = nil
    @State private var lvesAreaPixels: Int? = nil
    @State private var afcResultValue: Double? = nil

    // MARK: - Services
    private let videoProcessor = VideoProcessor()
    private let segmentationService = SegmentationService()
    private let efCalculator = EFCalculator()

    // Custom binding for the Slider
    private var currentFrameIndexBinding: Binding<Double> {
        Binding<Double>(
            get: { Double(self.currentFrameIndex) },
            set: { self.currentFrameIndex = Int($0) }
        )
    }

    // MARK: - Body
    var body: some View {
        NavigationView {
            VStack {
                VideoPickerView(selectedVideoURL: $selectedVideoURL, videoFileName: $videoFileName)
                    .onChange(of: selectedVideoURL) { newURL in
                        clearAllData()
                        guard let url = newURL else { return }
                        processVideoAndSegmentAllFrames(url: url)
                    }
                
                VideoDisplayView(
                    selectedVideoURL: $selectedVideoURL, // Still needed for fallback if no frames/mask
                    displayedFrame: displayedFrame,    // Pass the currently selected frame
                    displayedMask: displayedMask       // Pass the currently selected mask
                )
                
                frameSliderView() // Add the slider and frame count display
                
                statusAndErrorMessages()

                EjectionFractionView(ejectionFraction: $ejectionFractionString, calculateEFAction: {
                    initiateAFCCalculation()
                })
                .disabled(segmentationMasks.compactMap { $0 }.count < 2 || isCalculatingAFC || isSegmenting || isProcessingVideo)

                Spacer()
            }
            .navigationTitle("EchoAnalyzer")
            .padding()
        }
        .onChange(of: currentFrameIndex) { newIndex in
            updateDisplayedFrameAndMask(for: newIndex)
        }
        .onChange(of: segmentationMasks) { _ in // Also update when masks change (e.g. after segmentation)
            updateDisplayedFrameAndMask(for: currentFrameIndex)
        }
    }

    // MARK: - UI Components
    @ViewBuilder
    private func frameSliderView() -> some View {
        if !extractedFrames.isEmpty {
            VStack {
                Slider(
                    value: currentFrameIndexBinding,
                    in: 0...Double(max(0, extractedFrames.count - 1)),
                    step: 1
                ) {
                    Text("Frame Selector") // Accessibility label for the slider
                }
                .disabled(extractedFrames.isEmpty || isProcessingVideo || isSegmenting)
                .padding(.horizontal)

                Text("Frame: \(currentFrameIndex + 1) / \(extractedFrames.count)")
                    .font(.caption)
            }
            .padding(.bottom, 5)
        }
    }
    
    @ViewBuilder
    private func statusAndErrorMessages() -> some View {
        // ... (statusAndErrorMessages content remains the same as before)
        Group { 
            if isProcessingVideo {
                ProgressView("Extracting frames...")
            } else if isSegmenting { 
                ProgressView("Segmenting all frames...")
            } else if isCalculatingAFC { 
                ProgressView("Calculating AFC...")
            }
        }.padding(.vertical, 5)
        
        if let error = processingError {
            Text("Frame Extraction Error: \(error)").foregroundColor(.red).padding(.horizontal).font(.caption)
        }
        if let error = segmentationError {
            Text("Segmentation Error: \(error)").foregroundColor(.red).padding(.horizontal).font(.caption)
        }
        if let error = afcError {
            Text("AFC Error: \(error)").foregroundColor(.red).padding(.horizontal).font(.caption)
        }
        
        if !extractedFrames.isEmpty && !isProcessingVideo && !isSegmenting && !isCalculatingAFC {
            let validMasksCount = segmentationMasks.compactMap { $0 }.count
            Text("Extracted \(extractedFrames.count) frames. Successfully segmented \(validMasksCount) frames.")
                .font(.footnote)
                .padding(.vertical, 2)
            
            if validMasksCount < extractedFrames.count && extractedFrames.count > 0 && validMasksCount > 0 {
                 Text("Note: Not all frames were successfully segmented. AFC might be less accurate.")
                    .font(.caption).foregroundColor(.orange).padding(.horizontal)
            } else if validMasksCount == 0 && extractedFrames.count > 0 {
                Text("Warning: No frames were successfully segmented. Cannot calculate AFC.")
                    .font(.caption).foregroundColor(.red).padding(.horizontal)
            }
        }
    }
    
    // MARK: - Data Handling & Processing Flow
    private func clearAllData() {
        extractedFrames = []
        segmentationMasks = []
        selectedVideoURL = nil
        videoFileName = "No video selected"
        
        currentFrameIndex = 0
        displayedFrame = nil
        displayedMask = nil
        
        processingError = nil
        segmentationError = nil
        afcError = nil
        
        lvedAreaPixels = nil
        lvesAreaPixels = nil
        afcResultValue = nil
        ejectionFractionString = "--%"
        
        isProcessingVideo = false
        isSegmenting = false
        isCalculatingAFC = false
    }

    private func processVideoAndSegmentAllFrames(url: URL) {
        isProcessingVideo = true
        processingError = nil
        segmentationError = nil
        afcError = nil
        ejectionFractionString = "--%"
        currentFrameIndex = 0 // Reset index for new video

        videoProcessor.extractFrames(from: url, framesPerSecond: 5.0) { result in
            DispatchQueue.main.async {
                isProcessingVideo = false
                switch result {
                case .success(let frames):
                    self.extractedFrames = frames
                    if frames.isEmpty {
                        self.processingError = "No frames could be extracted from the video."
                        self.segmentationMasks = []
                        updateDisplayedFrameAndMask(for: 0) // Clear display
                    } else {
                        print("Successfully extracted \(frames.count) frames.")
                        // Initialize displayed frame right after extraction
                        updateDisplayedFrameAndMask(for: 0)
                        performSegmentationOnAllFrames(frames: frames)
                    }
                case .failure(let error):
                    self.processingError = error.localizedDescription
                    print("Failed to extract frames: \(error.localizedDescription)")
                    self.extractedFrames = []
                    self.segmentationMasks = []
                    updateDisplayedFrameAndMask(for: 0) // Clear display
                }
            }
        }
    }

    private func performSegmentationOnAllFrames(frames: [UIImage]) {
        guard !frames.isEmpty else {
            self.segmentationError = "No frames provided for segmentation."
            self.segmentationMasks = []
            updateDisplayedFrameAndMask(for: currentFrameIndex) // Update display, likely to show no mask
            return
        }
        isSegmenting = true
        self.segmentationMasks = Array(repeating: nil, count: frames.count)
        let dispatchGroup = DispatchGroup()

        for (index, frame) in frames.enumerated() {
            dispatchGroup.enter()
            Task {
                let (maskImage, _, error) = await segmentationService.segment(frame: frame)
                DispatchQueue.main.async {
                    if let err = error {
                         print("Segmentation failed for frame \(index): \(err.localizedDescription)")
                         if self.segmentationError == nil {
                             self.segmentationError = "Segmentation failed for one or more frames."
                         }
                    } else if let mi = maskImage {
                        self.segmentationMasks[index] = mi
                    } else {
                        print("Segmentation returned no image and no error for frame \(index).")
                        if self.segmentationError == nil {
                             self.segmentationError = "Segmentation returned no image for one or more frames."
                         }
                    }
                    // Update displayed mask if the current frame's mask was just processed
                    if index == self.currentFrameIndex {
                        updateDisplayedFrameAndMask(for: self.currentFrameIndex)
                    }
                    dispatchGroup.leave()
                }
            }
        }

        dispatchGroup.notify(queue: .main) {
            self.isSegmenting = false
            // Update display after all segmentations are done, in case initial display was nil
            updateDisplayedFrameAndMask(for: self.currentFrameIndex) 
            
            let successfulMasks = self.segmentationMasks.compactMap { $0 }.count
            print("Segmentation completed. Successfully segmented \(successfulMasks) of \(frames.count) frames.")
            if successfulMasks == 0 && !frames.isEmpty {
                self.segmentationError = (self.segmentationError ?? "") + " No frames were successfully segmented."
            } else if successfulMasks < frames.count && !frames.isEmpty {
                 self.segmentationError = (self.segmentationError ?? "") + " Some frames could not be segmented."
            }
            
            if successfulMasks >= 2 && !isCalculatingAFC {
                initiateAFCCalculation()
            } else if successfulMasks < 2 && !frames.isEmpty {
                self.afcError = "Not enough masks (\(successfulMasks)) for AFC. Need at least 2."
                self.ejectionFractionString = "Error: Low Masks"
            }
        }
    }
    
    private func initiateAFCCalculation() {
        // ... (initiateAFCCalculation content remains the same as before)
        let validMasks = segmentationMasks.compactMap { $0 }
        guard validMasks.count >= 2 else {
            self.afcError = "Not enough valid masks (\(validMasks.count)) for AFC calculation. Need at least 2."
            self.ejectionFractionString = "Error: Low Masks"
            return
        }
        
        isCalculatingAFC = true
        afcError = nil 
        
        let edEsResult = efCalculator.identifyEDandESFrames(masks: self.segmentationMasks) 
        
        guard case .success(let (edMask, esMask)) = edEsResult, let ed = edMask, let es = esMask else {
            if case .failure(let error) = edEsResult {
                self.afcError = "ED/ES ID failed: \(error.localizedDescription)"
            } else {
                self.afcError = "Could not identify ED and ES frames from available masks."
            }
            self.ejectionFractionString = "Error: ED/ES ID"
            isCalculatingAFC = false
            return
        }
        
        let edAreaResult = efCalculator.calculateLVArea(from: ed)
        let esAreaResult = efCalculator.calculateLVArea(from: es)

        guard case .success(let edArea) = edAreaResult, case .success(let esArea) = esAreaResult else {
            self.afcError = "Failed to calculate LV area for ED or ES frame. ED: \(edAreaResult), ES: \(esAreaResult)"
            self.ejectionFractionString = "Error: Area Calc"
            isCalculatingAFC = false
            return
        }
        self.lvedAreaPixels = edArea
        self.lvesAreaPixels = esArea
        
        let afcCalcResult = efCalculator.calculateAFC(lvedaPixels: edArea, lvesaPixels: esArea)
        
        switch afcCalcResult {
        case .success(let afc):
            self.afcResultValue = afc
            self.ejectionFractionString = String(format: "AFC: %.1f%%", afc)
            print("AFC Calculated: \(self.ejectionFractionString)")
        case .failure(let error):
            self.afcError = "AFC calculation error: \(error.localizedDescription)"
            self.ejectionFractionString = "Error: AFC Calc"
            print(self.afcError!)
        }
        isCalculatingAFC = false
    }

    private func updateDisplayedFrameAndMask(for index: Int) {
        guard !extractedFrames.isEmpty else {
            displayedFrame = nil
            displayedMask = nil
            return
        }
        // Ensure index is within bounds
        let safeIndex = max(0, min(index, extractedFrames.count - 1))
        
        displayedFrame = extractedFrames[safeIndex]
        
        if safeIndex < segmentationMasks.count {
            displayedMask = segmentationMasks[safeIndex]
        } else {
            displayedMask = nil // No corresponding mask if index is out of bounds for masks
        }
        
        // If the currentFrameIndex was adjusted, update it.
        if self.currentFrameIndex != safeIndex {
             self.currentFrameIndex = safeIndex
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
