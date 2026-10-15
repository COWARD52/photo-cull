import Foundation
import Vision
import CoreImage
import CoreGraphics
import UIKit

struct PhotoAnalysis {
    let sharpness: Double
    let exposure: Double
    let score: Int
    let reason: String
    let compositionScore: Int
    let compositionPositionScore: Int
    let compositionBalanceScore: Int
    let compositionEdgeScore: Int
    let compositionThirdsScore: Int
    let compositionTilt: Double
    let compositionNote: String
    let featurePrint: VNFeaturePrintObservation?
}

enum PhotoAnalyzer {
    static func analyze(_ image: UIImage) -> PhotoAnalysis {
        guard let cgImage = image.cgImage else {
            return PhotoAnalysis(sharpness: 0, exposure: 0, score: 50, reason: "无法读取预览",
                                 compositionScore: 50, compositionPositionScore: 50,
                                 compositionBalanceScore: 50, compositionEdgeScore: 50,
                                 compositionThirdsScore: 0, compositionTilt: 0,
                                 compositionNote: "构图参考不可用", featurePrint: nil)
        }
        let width = min(cgImage.width, 256)
        let height = max(1, Int(Double(cgImage.height) * Double(width) / Double(cgImage.width)))
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        pixels.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            context.interpolationQuality = .medium
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var gray = [Double](repeating: 0, count: width * height)
        var dark = 0
        var bright = 0
        for p in 0..<(width * height) {
            let value = 0.299 * Double(pixels[p * 4]) + 0.587 * Double(pixels[p * 4 + 1]) + 0.114 * Double(pixels[p * 4 + 2])
            gray[p] = value
            if value < 18 { dark += 1 }
            if value > 238 { bright += 1 }
        }
        let mean = gray.reduce(0, +) / Double(max(1, gray.count))
        var edgeEnergy = 0.0
        var samples = 0
        if width > 2 && height > 2 {
            for y in stride(from: 1, to: height - 1, by: 2) {
                for x in stride(from: 1, to: width - 1, by: 2) {
                    let p = y * width + x
                    let laplacian = 4 * gray[p] - gray[p - 1] - gray[p + 1] - gray[p - width] - gray[p + width]
                    edgeEnergy += laplacian * laplacian
                    samples += 1
                }
            }
        }
        let sharp = min(100, log10(edgeEnergy / Double(max(1, samples)) + 1) * 34)
        let extremes = Double(dark + bright) / Double(max(1, gray.count))
        let exposure = max(0, 100 - abs(mean - 128) * 0.78 - extremes * 90)
        let reason: String
        if sharp < 42 { reason = "边缘细节较少，可能轻微模糊" }
        else if mean < 54 || Double(dark) / Double(gray.count) > 0.55 { reason = "整体偏暗" }
        else if mean > 205 || Double(bright) / Double(gray.count) > 0.4 { reason = "高光区域较多" }
        else { reason = "清晰度与曝光表现较好" }

        let featurePrintRequest = VNGenerateImageFeaturePrintRequest()
        featurePrintRequest.imageCropAndScaleOption = .centerCrop
        let faceRequest = VNDetectFaceRectanglesRequest()
        try? VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([featurePrintRequest, faceRequest])
        let faces = (faceRequest.results ?? []).map(\.boundingBox).filter { $0.width > 0 && $0.height > 0 }
        let composition = compositionMetrics(gray: gray, width: width, height: height, faces: faces)
        let technicalScore = sharp * 0.62 + exposure * 0.38
        let score = max(1, min(99, Int((technicalScore * 0.88 + Double(composition.score) * 0.12).rounded())))
        return PhotoAnalysis(sharpness: sharp, exposure: exposure, score: score, reason: reason,
                             compositionScore: composition.score,
                             compositionPositionScore: composition.position,
                             compositionBalanceScore: composition.balance,
                             compositionEdgeScore: composition.edge,
                             compositionThirdsScore: composition.thirds,
                             compositionTilt: composition.tilt,
                             compositionNote: composition.note,
                             featurePrint: featurePrintRequest.results?.first)
    }

    // Estimate visual subject placement, balance, edge pressure and horizon tilt.
    // These are composition hints, not semantic understanding or an aesthetic verdict.
    private static func compositionMetrics(gray: [Double], width: Int, height: Int,
                                           faces: [CGRect]) -> (score: Int, position: Int, balance: Int,
                                                                edge: Int, thirds: Int, tilt: Double, note: String) {
        let sampleWidth = min(128, width)
        let sampleHeight = max(1, Int((Double(height) * Double(sampleWidth) / Double(width)).rounded()))
        var sample = [Double](repeating: 0, count: sampleWidth * sampleHeight)
        for y in 0..<sampleHeight {
            let sourceY = min(height - 1, y * height / sampleHeight)
            for x in 0..<sampleWidth {
                let sourceX = min(width - 1, x * width / sampleWidth)
                sample[y * sampleWidth + x] = gray[sourceY * width + sourceX]
            }
        }
        var saliency = [Double](repeating: 0, count: sample.count)
        var gradientsX = [Double](repeating: 0, count: sample.count)
        var gradientsY = [Double](repeating: 0, count: sample.count)
        if sampleWidth > 4 && sampleHeight > 4 {
            for y in 2..<(sampleHeight - 2) {
                for x in 2..<(sampleWidth - 2) {
                    let index = y * sampleWidth + x
                    var localSum = 0.0
                    for by in (y - 2)...(y + 2) {
                        for bx in (x - 2)...(x + 2) { localSum += sample[by * sampleWidth + bx] }
                    }
                    let gx = sample[index + 1] - sample[index - 1]
                    let gy = sample[index + sampleWidth] - sample[index - sampleWidth]
                    gradientsX[index] = gx
                    gradientsY[index] = gy
                    saliency[index] = abs(sample[index] - localSum / 25) + hypot(gx, gy) * 0.42
                }
            }
        }
        let sorted = saliency.sorted()
        let threshold = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.68))]
        var weights = [Double](repeating: 0, count: saliency.count)
        var total = 0.0, weightedX = 0.0, weightedY = 0.0
        for y in 0..<sampleHeight {
            for x in 0..<sampleWidth {
                let index = y * sampleWidth + x
                let weight = max(0, saliency[index] - threshold)
                weights[index] = weight
                total += weight
                weightedX += weight * Double(x) / Double(max(1, sampleWidth - 1))
                weightedY += weight * Double(y) / Double(max(1, sampleHeight - 1))
            }
        }
        if total < 1e-8 {
            total = 0
            weightedX = 0
            weightedY = 0
            let mean = sample.reduce(0, +) / Double(max(1, sample.count))
            for y in 0..<sampleHeight {
                for x in 0..<sampleWidth {
                    let index = y * sampleWidth + x
                    let weight = abs(sample[index] - mean) + 1e-3
                    weights[index] = weight
                    total += weight
                    weightedX += weight * Double(x) / Double(max(1, sampleWidth - 1))
                    weightedY += weight * Double(y) / Double(max(1, sampleHeight - 1))
                }
            }
        }
        let saliencyX = weightedX / max(total, 1e-8)
        let saliencyY = weightedY / max(total, 1e-8)
        let validFaces = faces.filter { $0.width * $0.height > 0 }
        let subjectX: Double
        let subjectY: Double
        if !validFaces.isEmpty {
            let areas = validFaces.map { max($0.width * $0.height, 1e-5) }
            let faceArea = areas.reduce(0, +)
            subjectX = zip(validFaces, areas).reduce(0) { $0 + Double($1.0.midX) * Double($1.1) } / Double(faceArea)
            subjectY = zip(validFaces, areas).reduce(0) { $0 + (1 - Double($1.0.midY)) * Double($1.1) } / Double(faceArea)
        } else {
            subjectX = saliencyX
            subjectY = saliencyY
        }
        let intersections = [(1.0 / 3, 1.0 / 3), (2.0 / 3, 1.0 / 3), (1.0 / 3, 2.0 / 3), (2.0 / 3, 2.0 / 3)]
        let thirdsDistance = intersections.map { hypot(subjectX - $0.0, subjectY - $0.1) }.min() ?? 1
        let thirdsScore = Int((100 * exp(-2.2 * pow(thirdsDistance / 0.36, 2))).rounded())
        let anchors: [(Double, Double)] = validFaces.isEmpty
            ? [(0.5, 0.5), (1.0 / 3, 1.0 / 3), (2.0 / 3, 1.0 / 3), (1.0 / 3, 2.0 / 3), (2.0 / 3, 2.0 / 3)]
            : [(0.5, 0.38), (1.0 / 3, 0.36), (2.0 / 3, 0.36)]
        let normalizedDistance = anchors.map { hypot((subjectX - $0.0) / 0.42, (subjectY - $0.1) / 0.42) }.min() ?? 1
        let positionScore = Int(max(20, 100 * exp(-1.45 * normalizedDistance * normalizedDistance)).rounded())

        let halfWidth = sampleWidth / 2
        let halfHeight = sampleHeight / 2
        let left = weights.indices.filter { $0 % sampleWidth < halfWidth }.reduce(0) { $0 + weights[$1] } / max(total, 1e-8)
        let right = weights.indices.filter { $0 % sampleWidth >= halfWidth }.reduce(0) { $0 + weights[$1] } / max(total, 1e-8)
        let top = weights.indices.filter { $0 / sampleWidth < halfHeight }.reduce(0) { $0 + weights[$1] } / max(total, 1e-8)
        let bottom = weights.indices.filter { $0 / sampleWidth >= halfHeight }.reduce(0) { $0 + weights[$1] } / max(total, 1e-8)
        let horizontalImbalance = abs(left - right)
        let verticalImbalance = abs(top - bottom)
        let horizontalPenalty = 72.0 * horizontalImbalance
        let verticalPenalty = 38.0 * verticalImbalance
        let rawBalanceScore = 100.0 - horizontalPenalty - verticalPenalty
        let boundedBalanceScore = max(20.0, rawBalanceScore)
        let balanceScore = Int(boundedBalanceScore.rounded())

        let margin = max(4, Int((Double(min(sampleWidth, sampleHeight)) * 0.075).rounded()))
        var borderWeight = 0.0
        for y in 0..<sampleHeight {
            for x in 0..<sampleWidth where x < margin || x >= sampleWidth - margin || y < margin || y >= sampleHeight - margin {
                borderWeight += weights[y * sampleWidth + x]
            }
        }
        var edgeScore = max(20, Int((100 - max(0, borderWeight / max(total, 1e-8) - 0.13) * 190).rounded()))
        let faceTouching = validFaces.contains { face in
            let clearance = [face.minX, face.minY, 1 - face.maxX, 1 - face.maxY].min() ?? 1
            return clearance < 0.009
        }
        let faceClose = validFaces.contains { face in
            let clearance = [face.minX, face.minY, 1 - face.maxX, 1 - face.maxY].min() ?? 1
            return clearance < 0.028
        }
        if faceTouching { edgeScore = min(edgeScore, 34) }
        else if faceClose { edgeScore = min(edgeScore, 58) }

        let gradientMagnitudes = gradientsX.indices.map { hypot(gradientsX[$0], gradientsY[$0]) }.sorted()
        let strongThreshold = gradientMagnitudes[min(gradientMagnitudes.count - 1, Int(Double(gradientMagnitudes.count - 1) * 0.86))]
        var strongWeight = 0.0, horizontalWeight = 0.0, tiltSin = 0.0, tiltCos = 0.0
        for index in weights.indices where hypot(gradientsX[index], gradientsY[index]) >= strongThreshold {
            let gx = gradientsX[index], gy = gradientsY[index]
            let magnitude = hypot(gx, gy)
            guard magnitude > 0 else { continue }
            strongWeight += magnitude
            var lineAngle = (atan2(gy, gx) + .pi / 2) * 180 / .pi
            lineAngle = (lineAngle + 90).truncatingRemainder(dividingBy: 180) - 90
            if abs(lineAngle) <= 18 {
                let radians = lineAngle * .pi / 90
                horizontalWeight += magnitude
                tiltSin += sin(radians) * magnitude
                tiltCos += cos(radians) * magnitude
            }
        }
        var tilt = 0.0
        if horizontalWeight / max(strongWeight, 1e-8) >= 0.2 && horizontalWeight > 0 {
            tilt = 0.5 * atan2(tiltSin, tiltCos) * 180 / .pi
        }
        let tiltPenalty = min(12, max(0, abs(tilt) - 2.2) * 1.6)
        let compositionScore = Int(max(1, min(100, Double(positionScore) * 0.45 + Double(balanceScore) * 0.30 + Double(edgeScore) * 0.25 - tiltPenalty)).rounded())
        let horizontalName = subjectX < 0.42 ? "左侧" : (subjectX > 0.58 ? "右侧" : "中央")
        let verticalName = subjectY < 0.4 ? "上方" : (subjectY > 0.64 ? "下方" : "中部")
        let subjectName = validFaces.isEmpty ? "视觉主体" : "人物主体"
        let tiltText = String(format: "%.1f", abs(tilt))
        let note = "\(subjectName)约在\(horizontalName)\(verticalName)；平衡\(balanceScore)；边缘安全\(edgeScore)；水平线倾斜约\(tiltText)°"
        return (compositionScore, positionScore, balanceScore, edgeScore, thirdsScore, tilt, note)
    }

    static func similarity(_ lhs: VNFeaturePrintObservation?, _ rhs: VNFeaturePrintObservation?) -> Float? {
        guard let lhs, let rhs else { return nil }
        var distance: Float = 0
        do { try lhs.computeDistance(&distance, to: rhs); return distance }
        catch { return nil }
    }
}
