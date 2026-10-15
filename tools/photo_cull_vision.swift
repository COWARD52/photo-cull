import Foundation
import Vision

struct FaceResult: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    let leftEyeRatio: Double?
    let rightEyeRatio: Double?
}

struct ImageResult: Codable {
    let path: String
    let faces: [FaceResult]
}

func eyeRatio(_ region: VNFaceLandmarkRegion2D?) -> Double? {
    guard let region, region.pointCount >= 4 else { return nil }
    let points = region.normalizedPoints
    let xs = points.map { Double($0.x) }
    let ys = points.map { Double($0.y) }
    guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return nil }
    let width = max(maxX - minX, 0.0001)
    return (maxY - minY) / width
}

let input = FileHandle.standardInput.readDataToEndOfFile()
guard let paths = try? JSONDecoder().decode([String].self, from: input) else {
    FileHandle.standardError.write(Data("invalid input\n".utf8))
    exit(2)
}

var output: [ImageResult] = []
for path in paths {
    let request = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
    do {
        try handler.perform([request])
        let faces = (request.results ?? []).map { face in
            FaceResult(
                x: Double(face.boundingBox.origin.x),
                y: Double(face.boundingBox.origin.y),
                width: Double(face.boundingBox.size.width),
                height: Double(face.boundingBox.size.height),
                leftEyeRatio: eyeRatio(face.landmarks?.leftEye),
                rightEyeRatio: eyeRatio(face.landmarks?.rightEye)
            )
        }
        output.append(ImageResult(path: path, faces: faces))
    } catch {
        output.append(ImageResult(path: path, faces: []))
    }
}

let encoder = JSONEncoder()
if let data = try? encoder.encode(output) {
    FileHandle.standardOutput.write(data)
} else {
    exit(3)
}
