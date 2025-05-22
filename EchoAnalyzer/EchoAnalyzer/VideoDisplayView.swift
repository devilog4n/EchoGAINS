import SwiftUI
import AVKit // For AVPlayer
import UIKit // For UIImage

struct VideoDisplayView: View {
    @Binding var selectedVideoURL: URL? // Keep for fallback if no frame is ready
    var displayedFrame: UIImage?         // Renamed from firstFrame
    var displayedMask: UIImage?          // Renamed from segmentationMask

    var body: some View {
        VStack {
            Text("Video Display Area")
                .font(.headline)
                .padding(.bottom, 5)

            ZStack {
                // Display the current frame
                if let frame = displayedFrame {
                    Image(uiImage: frame)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                        .cornerRadius(10)
                        .accessibilityLabel("Currently selected video frame")
                } 
                // Fallback if no specific frame is displayed (e.g., initial state or error)
                // but a video URL is selected. This provides some visual continuity.
                else if let url = selectedVideoURL { 
                    VideoPlayer(player: AVPlayer(url: url))
                        .frame(height: 300)
                        .cornerRadius(10)
                        .accessibilityLabel("Video player displaying the selected echocardiogram")
                        .disabled(true) // Disable interaction if it's just a fallback
                } 
                // Placeholder if nothing is available
                else {
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(height: 300)
                        .cornerRadius(10)
                        .overlay(
                            Text("Video will be displayed here")
                                .foregroundColor(.gray)
                        )
                        .accessibilityLabel("Placeholder for video display area")
                }

                // Overlay the segmentation mask for the current frame if available
                if let mask = displayedMask {
                    Image(uiImage: mask)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                        .cornerRadius(10)
                        .opacity(0.7) 
                        .accessibilityLabel("Segmentation mask for the current frame")
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.gray, lineWidth: 1)
            )
            
            // Informative text if a frame is shown but its specific mask isn't available
            if displayedFrame != nil && displayedMask == nil {
                Text("Segmentation mask not available for this frame.")
                    .font(.caption)
                    .foregroundColor(.orange)
                    .padding(.top, 5)
            }
        }
        .padding()
    }
}

// Preview
struct VideoDisplayView_Previews: PreviewProvider {
    @State static var previewSelectedURL: URL? = nil // Not directly used by displayedFrame/Mask
    
    static var sampleFrame: UIImage = {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 320, height: 240))
        return renderer.image { context in
            UIColor.systemBlue.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 320, height: 240))
            ("Sample Frame" as NSString).draw(at: CGPoint(x: 80, y: 100), 
                                            withAttributes: [.font: UIFont.systemFont(ofSize: 24), .foregroundColor: UIColor.white])
        }
    }()
    
    static var sampleMask: UIImage? = { // Make it optional for one preview case
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 320, height: 240))
        return renderer.image { context in
            UIColor.red.withAlphaComponent(0.5).setFill()
            UIBezierPath(ovalIn: CGRect(x: 80, y: 60, width: 160, height: 120)).fill()
        }
    }()

    static var previews: some View {
        Group {
            VideoDisplayView(selectedVideoURL: .constant(nil), displayedFrame: nil, displayedMask: nil)
                .previewDisplayName("No Frame, No Mask")
            
            VideoDisplayView(selectedVideoURL: .constant(nil), displayedFrame: sampleFrame, displayedMask: nil)
                .previewDisplayName("Frame, No Mask")

            VideoDisplayView(selectedVideoURL: .constant(nil), displayedFrame: sampleFrame, displayedMask: sampleMask)
                .previewDisplayName("Frame with Mask")
            
            // Example with a placeholder URL (fallback, frame/mask are nil)
            VideoDisplayView(selectedVideoURL: .constant(URL(string: "file:///sample.mp4")!), displayedFrame: nil, displayedMask: nil)
               .previewDisplayName("Video URL Fallback")
        }
        .previewLayout(.sizeThatFits)
        .padding()
    }
}
