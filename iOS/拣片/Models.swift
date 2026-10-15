import Foundation
import Photos
import UIKit

enum PhotoDecision: String, Codable, CaseIterable {
    case keep, review, reject

    var title: String {
        switch self {
        case .keep: "保留"
        case .review: "待确认"
        case .reject: "淘汰"
        }
    }
}

struct ScoredPhoto: Identifiable {
    let id: String
    let asset: PHAsset
    var thumbnail: UIImage?
    var score: Int
    var sharpness: Int
    var exposure: Int
    var reason: String
    var compositionScore: Int
    var compositionPositionScore: Int
    var compositionBalanceScore: Int
    var compositionEdgeScore: Int
    var compositionThirdsScore: Int
    var compositionTilt: Double
    var compositionNote: String
    var groupID: Int?
    var groupRank: Int
    var pairID: String?
    var decision: PhotoDecision
    var hash: UInt64

    var creationDate: Date { asset.creationDate ?? .distantPast }
    var filename: String {
        PHAssetResource.assetResources(for: asset).first?.originalFilename ?? "照片"
    }
}

struct SavedCullingSession: Codable {
    var albumID: String
    var decisions: [String: PhotoDecision]
    var updatedAt: Date
    // Optional so sessions saved by earlier app versions keep their decisions.
    var analysisVersion: Int?
    var analysis: [String: CachedPhotoAnalysis]?
}

struct CachedPhotoAnalysis: Codable {
    var assetSignature: String
    var score: Int
    var sharpness: Int
    var exposure: Int
    var reason: String
    var compositionScore: Int?
    var compositionPositionScore: Int?
    var compositionBalanceScore: Int?
    var compositionEdgeScore: Int?
    var compositionThirdsScore: Int?
    var compositionTilt: Double?
    var compositionNote: String?
    var groupID: Int?
    var groupRank: Int
    var pairID: String?
    var hash: UInt64
}

struct AlbumSummary: Identifiable {
    let id: String
    let title: String
    let count: Int
    let collection: PHAssetCollection
}
