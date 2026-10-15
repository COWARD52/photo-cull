import Foundation
import Combine
import Photos
import Vision
import UIKit
import CoreGraphics

@MainActor
final class CullingSession: ObservableObject {
    @Published var photos: [ScoredPhoto] = []
    @Published var selectedAlbum: AlbumSummary?
    @Published var isAnalyzing = false
    @Published var progress = 0.0
    @Published var statusText = "选择一个相册开始"
    @Published var filter: PhotoDecision?
    @Published var message: String?

    private let persistenceKey = "jianpian.ios.session.v1"
    private static let analysisCacheVersion = 3
    private var featurePrints: [String: VNFeaturePrintObservation] = [:]
    private var activeRunID: UUID?

    var keptCount: Int { photos.filter { $0.decision == .keep }.count }
    var reviewCount: Int { photos.filter { $0.decision == .review }.count }
    var rejectedCount: Int { photos.filter { $0.decision == .reject }.count }

    func start(album: AlbumSummary, library: PhotoLibraryStore) async {
        let runID = UUID()
        activeRunID = runID
        selectedAlbum = album
        isAnalyzing = true
        progress = 0
        statusText = "正在读取照片…"
        photos = []
        featurePrints = [:]
        let assets = library.assets(in: album.collection)
        guard !assets.isEmpty else {
            statusText = "这个相册里没有照片"
            isAnalyzing = false
            return
        }
        let saved = loadSavedSession(for: album.id)
        let cachedAnalysis = saved?.analysis
        let canReuseAnalysis = saved?.analysisVersion == Self.analysisCacheVersion
            && cachedAnalysis?.count == assets.count
            && assets.allSatisfy { asset in
                cachedAnalysis?[asset.localIdentifier]?.assetSignature == Self.assetSignature(for: asset)
            }
        statusText = canReuseAnalysis ? "正在恢复分析结果并读取预览…" : "正在分析照片…"

        for (index, asset) in assets.enumerated() {
            guard activeRunID == runID else { return }
            statusText = canReuseAnalysis
                ? "正在恢复第 \(index + 1) / \(assets.count) 张"
                : "正在分析第 \(index + 1) / \(assets.count) 张"
            guard let image = await library.thumbnail(for: asset) else {
                progress = Double(index + 1) / Double(assets.count)
                continue
            }
            if canReuseAnalysis, let cached = cachedAnalysis?[asset.localIdentifier] {
                let decision = saved?.decisions[asset.localIdentifier] ?? defaultDecision(score: cached.score)
                photos.append(ScoredPhoto(id: asset.localIdentifier, asset: asset, thumbnail: image,
                                          score: cached.score, sharpness: cached.sharpness,
                                          exposure: cached.exposure, reason: cached.reason,
                                          compositionScore: cached.compositionScore ?? 50,
                                          compositionPositionScore: cached.compositionPositionScore ?? 50,
                                          compositionBalanceScore: cached.compositionBalanceScore ?? 50,
                                          compositionEdgeScore: cached.compositionEdgeScore ?? 50,
                                          compositionThirdsScore: cached.compositionThirdsScore ?? 0,
                                          compositionTilt: cached.compositionTilt ?? 0,
                                          compositionNote: cached.compositionNote ?? "构图参考不可用",
                                          groupID: cached.groupID, groupRank: cached.groupRank,
                                          pairID: cached.pairID, decision: decision, hash: cached.hash))
            } else {
                let analysis = await Task.detached(priority: .utility) { PhotoAnalyzer.analyze(image) }.value
                guard activeRunID == runID else { return }
                let decision = saved?.decisions[asset.localIdentifier] ?? defaultDecision(score: analysis.score)
                photos.append(ScoredPhoto(id: asset.localIdentifier, asset: asset, thumbnail: image,
                                          score: analysis.score, sharpness: Int(analysis.sharpness.rounded()),
                                          exposure: Int(analysis.exposure.rounded()), reason: analysis.reason,
                                          compositionScore: analysis.compositionScore,
                                          compositionPositionScore: analysis.compositionPositionScore,
                                          compositionBalanceScore: analysis.compositionBalanceScore,
                                          compositionEdgeScore: analysis.compositionEdgeScore,
                                          compositionThirdsScore: analysis.compositionThirdsScore,
                                          compositionTilt: analysis.compositionTilt,
                                          compositionNote: analysis.compositionNote,
                                          groupID: nil, groupRank: 1, pairID: nil, decision: decision,
                                          hash: Self.perceptualHash(image)))
                if let print = analysis.featurePrint { featurePrints[asset.localIdentifier] = print }
            }
            progress = Double(index + 1) / Double(assets.count)
            if index.isMultiple(of: 4) { await Task.yield() }
        }
        guard activeRunID == runID else { return }
        if !canReuseAnalysis {
            linkRawJpegPairs()
            for index in photos.indices {
                guard let pairID = photos[index].pairID,
                      let best = photos.indices.filter({ photos[$0].pairID == pairID }).max(by: { photos[$0].score < photos[$1].score }) else { continue }
                photos[index].decision = photos[best].decision
            }
            makeSimilarityGroups()
            for index in photos.indices {
                guard let pairID = photos[index].pairID else { continue }
                let pair = photos.indices.filter { photos[$0].pairID == pairID }
                let decision: PhotoDecision = pair.contains(where: { photos[$0].decision == .reject }) ? .reject
                    : pair.contains(where: { photos[$0].decision == .review }) ? .review : .keep
                for pairedIndex in pair { photos[pairedIndex].decision = decision }
            }
        }
        persist()
        isAnalyzing = false
        statusText = canReuseAnalysis
            ? "已恢复缓存 · \(photos.count) 张照片"
            : "分析完成 · \(photos.count) 张照片"
    }

    func decide(_ photo: ScoredPhoto, as decision: PhotoDecision) {
        guard let index = photos.firstIndex(where: { $0.id == photo.id }) else { return }
        applyDecision(at: index, decision: decision)
        persist()
    }

    func setGroupDecision(_ photo: ScoredPhoto, as decision: PhotoDecision) {
        guard let group = photo.groupID else { decide(photo, as: decision); return }
        let groupIDs = Set(photos.filter { $0.groupID == group }.map(\.id))
        for index in photos.indices where groupIDs.contains(photos[index].id) { applyDecision(at: index, decision: decision) }
        persist()
    }

    func saveToAlbums(library: PhotoLibraryStore) async {
        guard let selectedAlbum else { return }
        do {
            message = try await library.organize(photos: photos, sourceAlbum: selectedAlbum)
        } catch {
            message = "整理失败：\(error.localizedDescription)"
        }
    }

    func leaveAlbum() {
        activeRunID = nil
        selectedAlbum = nil
        photos = []
        filter = nil
        isAnalyzing = false
        progress = 0
        statusText = "选择一个相册开始"
    }

    func visiblePhotos() -> [ScoredPhoto] {
        let filtered = filter.map { decision in photos.filter { $0.decision == decision } } ?? photos
        return filtered.sorted { $0.score > $1.score }
    }

    private func defaultDecision(score: Int) -> PhotoDecision {
        score >= 70 ? .keep : (score < 38 ? .reject : .review)
    }

    private func applyDecision(at index: Int, decision: PhotoDecision) {
        photos[index].decision = decision
        guard let pairID = photos[index].pairID else { return }
        for pairedIndex in photos.indices where photos[pairedIndex].pairID == pairID {
            photos[pairedIndex].decision = decision
        }
    }

    private func makeSimilarityGroups() {
        for index in photos.indices { photos[index].groupID = nil; photos[index].groupRank = 1 }
        let ordered = photos.indices.sorted { photos[$0].creationDate < photos[$1].creationDate }
        var nextGroupID = 0
        var assigned = Set<Int>()
        for (position, firstIndex) in ordered.enumerated() where !assigned.contains(firstIndex) {
            var group = [firstIndex]
            if position + 1 < ordered.count {
                for nextPosition in (position + 1)..<ordered.count {
                    let candidateIndex = ordered[nextPosition]
                    let gap = photos[candidateIndex].creationDate.timeIntervalSince(photos[firstIndex].creationDate)
                    if gap > 300 { break }
                    guard !assigned.contains(candidateIndex) else { continue }
                    if let distance = PhotoAnalyzer.similarity(featurePrints[photos[firstIndex].id], featurePrints[photos[candidateIndex].id]), distance < 0.32 {
                        group.append(candidateIndex)
                    }
                }
            }
            guard group.count > 1 else { continue }
            nextGroupID += 1
            group.sort { photos[$0].score > photos[$1].score }
            for (rank, photoIndex) in group.enumerated() {
                photos[photoIndex].groupID = nextGroupID
                photos[photoIndex].groupRank = rank + 1
                assigned.insert(photoIndex)
                if rank > 0 && photos[photoIndex].decision == .keep { photos[photoIndex].decision = .review }
            }
        }
    }

    private func linkRawJpegPairs() {
        var unmatched: [String: (index: Int, ext: String)] = [:]
        let rawExtensions: Set<String> = ["ARW", "CR2", "CR3", "NEF", "NRW", "RAF", "ORF", "RW2", "DNG", "PEF"]
        for index in photos.indices {
            let url = URL(fileURLWithPath: photos[index].filename)
            let ext = url.pathExtension.uppercased()
            let isRaw = rawExtensions.contains(ext)
            let isJpeg = ["JPG", "JPEG"].contains(ext)
            guard isRaw || isJpeg else { continue }
            let key = url.deletingPathExtension().lastPathComponent.uppercased()
            if let candidate = unmatched[key], candidate.ext != ext,
               abs(photos[candidate.index].creationDate.timeIntervalSince(photos[index].creationDate)) <= 3 {
                let pairID = "pair-\(key)-\(Int(photos[index].creationDate.timeIntervalSince1970))"
                photos[index].pairID = pairID
                photos[candidate.index].pairID = pairID
                unmatched.removeValue(forKey: key)
            } else {
                unmatched[key] = (index, ext)
            }
        }
    }

    private func persist() {
        guard let selectedAlbum else { return }
        let decisions = Dictionary(uniqueKeysWithValues: photos.map { ($0.id, $0.decision) })
        let analysis = Dictionary(uniqueKeysWithValues: photos.map { photo in
            (photo.id, CachedPhotoAnalysis(assetSignature: Self.assetSignature(for: photo.asset),
                                           score: photo.score, sharpness: photo.sharpness,
                                           exposure: photo.exposure, reason: photo.reason,
                                           compositionScore: photo.compositionScore,
                                           compositionPositionScore: photo.compositionPositionScore,
                                           compositionBalanceScore: photo.compositionBalanceScore,
                                           compositionEdgeScore: photo.compositionEdgeScore,
                                           compositionThirdsScore: photo.compositionThirdsScore,
                                           compositionTilt: photo.compositionTilt,
                                           compositionNote: photo.compositionNote,
                                           groupID: photo.groupID, groupRank: photo.groupRank,
                                           pairID: photo.pairID, hash: photo.hash))
        })
        let value = SavedCullingSession(albumID: selectedAlbum.id, decisions: decisions,
                                        updatedAt: Date(), analysisVersion: Self.analysisCacheVersion,
                                        analysis: analysis)
        if let data = try? JSONEncoder().encode(value) {
            UserDefaults.standard.set(data, forKey: sessionKey(for: selectedAlbum.id))
        }
    }

    private static func assetSignature(for asset: PHAsset) -> String {
        let modified = asset.modificationDate.map { String(Int64($0.timeIntervalSince1970 * 1_000)) } ?? "none"
        let created = asset.creationDate.map { String(Int64($0.timeIntervalSince1970 * 1_000)) } ?? "none"
        return "\(modified)|\(created)|\(asset.pixelWidth)x\(asset.pixelHeight)|\(asset.mediaSubtypes.rawValue)"
    }

    private func loadSavedSession(for albumID: String) -> SavedCullingSession? {
        let decoder = JSONDecoder()
        if let data = UserDefaults.standard.data(forKey: sessionKey(for: albumID)),
           let value = try? decoder.decode(SavedCullingSession.self, from: data), value.albumID == albumID {
            return value
        }
        // Migrate the single-session format used by older versions on next save.
        guard let data = UserDefaults.standard.data(forKey: persistenceKey),
              let value = try? decoder.decode(SavedCullingSession.self, from: data), value.albumID == albumID else { return nil }
        return value
    }

    private func sessionKey(for albumID: String) -> String { "\(persistenceKey).\(albumID)" }

    private static func perceptualHash(_ image: UIImage) -> UInt64 {
        guard let cgImage = image.cgImage else { return 0 }
        let width = 9, height = 8
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        pixels.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var hash: UInt64 = 0
        for y in 0..<height {
            for x in 0..<8 {
                let left = y * width + x, right = left + 1
                let l = UInt16(pixels[left * 4]) + UInt16(pixels[left * 4 + 1]) + UInt16(pixels[left * 4 + 2])
                let r = UInt16(pixels[right * 4]) + UInt16(pixels[right * 4 + 1]) + UInt16(pixels[right * 4 + 2])
                if l > r { hash |= 1 << UInt64(y * 8 + x) }
            }
        }
        return hash
    }
}
