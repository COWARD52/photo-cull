import Foundation
import Combine
import Photos
import UIKit

struct OrganizationReceipt {
    let sourceAlbum: PHAssetCollection
    let sourceAlbumTitle: String
    let sourceWasUserAlbum: Bool
    let reviewAddedIDs: [String]
    let rejectedAddedIDs: [String]
    let sourceRemovedIDs: [String]
}

@MainActor
final class PhotoLibraryStore: ObservableObject {
    @Published var authorization: PHAuthorizationStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published var albums: [AlbumSummary] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published private(set) var lastOrganizationReceipt: OrganizationReceipt?

    func requestAccess() async {
        authorization = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        if authorization == .authorized || authorization == .limited { loadAlbums() }
    }

    func loadAlbums() {
        guard authorization == .authorized || authorization == .limited else { return }
        isLoading = true
        var results: [AlbumSummary] = []

        let smart = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .any, options: nil)
        smart.enumerateObjects { collection, _, _ in
            guard collection.assetCollectionSubtype == .smartAlbumUserLibrary else { return }
            let count = PHAsset.fetchAssets(in: collection, options: nil).count
            results.append(AlbumSummary(id: collection.localIdentifier, title: "最近项目", count: count, collection: collection))
        }
        let user = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: nil)
        user.enumerateObjects { collection, _, _ in
            let count = PHAsset.fetchAssets(in: collection, options: nil).count
            results.append(AlbumSummary(id: collection.localIdentifier, title: collection.localizedTitle ?? "未命名相册", count: count, collection: collection))
        }
        albums = results.sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
        isLoading = false
    }

    func assets(in collection: PHAssetCollection) -> [PHAsset] {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: true)]
        let fetched = PHAsset.fetchAssets(in: collection, options: options)
        var assets: [PHAsset] = []
        fetched.enumerateObjects { asset, _, _ in
            if asset.mediaType == .image { assets.append(asset) }
        }
        return assets
    }

    // Keep the retained per-photo previews small. CullingSession holds one preview
    // for every asset so the grid can render instantly; 900px decoded images can
    // consume hundreds of MB for a typical camera-roll album and get the app killed.
    func thumbnail(for asset: PHAsset, size: CGSize = CGSize(width: 480, height: 480)) async -> UIImage? {
        await withCheckedContinuation { continuation in
            let options = PHImageRequestOptions()
            options.deliveryMode = .highQualityFormat
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = true
            PHImageManager.default().requestImage(for: asset, targetSize: size, contentMode: .aspectFit, options: options) { image, info in
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
                continuation.resume(returning: image)
            }
        }
    }

    func organize(photos: [ScoredPhoto], sourceAlbum: AlbumSummary) async throws -> String {
        let reviewAssets = photos.filter { $0.decision == .review }.map(\.asset)
        let rejectedAssets = photos.filter { $0.decision == .reject }.map(\.asset)
        lastOrganizationReceipt = nil
        let buckets: [(title: String, assets: [PHAsset])] = [
            ("\(sourceAlbum.title)_待确认", reviewAssets),
            ("\(sourceAlbum.title)_待删除", rejectedAssets)
        ].filter { !$0.assets.isEmpty }
        guard !buckets.isEmpty else {
            return "没有待确认或待删除照片；保留照片仍留在“\(sourceAlbum.title)”中。"
        }

        let existingAlbums = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: nil)
        var targets: [String: PHAssetCollection] = [:]
        for bucket in buckets {
            existingAlbums.enumerateObjects { collection, _, stop in
                if collection.localizedTitle == bucket.title {
                    targets[bucket.title] = collection
                    stop.pointee = true
                }
            }
        }
        var existingMemberIDs: [String: Set<String>] = [:]
        for (title, album) in targets {
            let fetched = PHAsset.fetchAssets(in: album, options: nil)
            var ids = Set<String>()
            fetched.enumerateObjects { asset, _, _ in ids.insert(asset.localIdentifier) }
            existingMemberIDs[title] = ids
        }

        let shouldMoveOutOfSource = sourceAlbum.collection.assetCollectionType == .album
        let assetsToRemove = reviewAssets + rejectedAssets
        var reviewAddedIDs: [String] = []
        var rejectedAddedIDs: [String] = []
        try await PHPhotoLibrary.shared().performChanges {
            for bucket in buckets {
                let alreadyPresent = existingMemberIDs[bucket.title] ?? []
                let newAssets = bucket.assets.filter { !alreadyPresent.contains($0.localIdentifier) }
                guard !newAssets.isEmpty else { continue }
                if bucket.title == "\(sourceAlbum.title)_待确认" {
                    reviewAddedIDs = newAssets.map(\.localIdentifier)
                } else {
                    rejectedAddedIDs = newAssets.map(\.localIdentifier)
                }
                if let target = targets[bucket.title] {
                    PHAssetCollectionChangeRequest(for: target)?.addAssets(newAssets as NSArray)
                } else {
                    let change = PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle: bucket.title)
                    change.addAssets(newAssets as NSArray)
                }
            }
            if shouldMoveOutOfSource, let sourceChange = PHAssetCollectionChangeRequest(for: sourceAlbum.collection) {
                sourceChange.removeAssets(assetsToRemove as NSArray)
            }
        }

        loadAlbums()
        lastOrganizationReceipt = OrganizationReceipt(
            sourceAlbum: sourceAlbum.collection,
            sourceAlbumTitle: sourceAlbum.title,
            sourceWasUserAlbum: shouldMoveOutOfSource,
            reviewAddedIDs: reviewAddedIDs,
            rejectedAddedIDs: rejectedAddedIDs,
            sourceRemovedIDs: shouldMoveOutOfSource ? assetsToRemove.map(\.localIdentifier) : []
        )
        let movedText = shouldMoveOutOfSource ? "已从来源相册移出" : "系统智能相册无法移出原片，已加入目标相册"
        return "整理完成：\(reviewAssets.count) 张进入“\(sourceAlbum.title)_待确认”，\(rejectedAssets.count) 张进入“\(sourceAlbum.title)_待删除”；\(movedText)。保留照片留在原相册，照片仍在系统图库中，未被删除。"
    }

    func undoLastOrganization() async throws -> String {
        guard let receipt = lastOrganizationReceipt else { throw PhotoLibraryError.nothingToUndo }
        let reviewAssets = PHAsset.fetchAssets(withLocalIdentifiers: receipt.reviewAddedIDs, options: nil)
        let rejectedAssets = PHAsset.fetchAssets(withLocalIdentifiers: receipt.rejectedAddedIDs, options: nil)
        let reviewAlbum = findAlbum(named: "\(receipt.sourceAlbumTitle)_待确认")
        let rejectedAlbum = findAlbum(named: "\(receipt.sourceAlbumTitle)_待删除")
        let reviewArray = assetsArray(reviewAssets)
        let rejectedArray = assetsArray(rejectedAssets)

        try await PHPhotoLibrary.shared().performChanges {
            if let reviewAlbum, reviewArray.count > 0 {
                PHAssetCollectionChangeRequest(for: reviewAlbum)?.removeAssets(reviewArray)
            }
            if let rejectedAlbum, rejectedArray.count > 0 {
                PHAssetCollectionChangeRequest(for: rejectedAlbum)?.removeAssets(rejectedArray)
            }
            if receipt.sourceWasUserAlbum {
                let sourceAssets = self.assetsArray(PHAsset.fetchAssets(withLocalIdentifiers: receipt.sourceRemovedIDs, options: nil))
                if sourceAssets.count > 0 {
                    PHAssetCollectionChangeRequest(for: receipt.sourceAlbum)?.addAssets(sourceAssets)
                }
            }
        }

        lastOrganizationReceipt = nil
        loadAlbums()
        return receipt.sourceWasUserAlbum
            ? "已撤销整理：照片已从结果相册移除，并放回来源相册。"
            : "已撤销整理：照片已从结果相册移除；智能相册中的原片未移动。"
    }

    private func findAlbum(named title: String) -> PHAssetCollection? {
        let albums = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: nil)
        var match: PHAssetCollection?
        albums.enumerateObjects { album, _, stop in
            if album.localizedTitle == title {
                match = album
                stop.pointee = true
            }
        }
        return match
    }

    private func assetsArray(_ result: PHFetchResult<PHAsset>) -> NSArray {
        let assets = NSMutableArray(capacity: result.count)
        result.enumerateObjects { asset, _, _ in assets.add(asset) }
        return assets
    }
}

private enum PhotoLibraryError: LocalizedError {
    case nothingToUndo

    var errorDescription: String? {
        "当前没有可以撤销的整理操作。"
    }
}
