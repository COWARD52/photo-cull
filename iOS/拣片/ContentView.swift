import SwiftUI
import Photos
import Security

struct ContentView: View {
    @EnvironmentObject private var library: PhotoLibraryStore
    @EnvironmentObject private var session: CullingSession
    @State private var activeHomeSheet: HomeSheet?
    @State private var showSavedMessage = false
    @State private var showOrganizeConfirmation = false
    @State private var isOrganizing = false
    @State private var presentedScreen: ReviewScreen?
    @State private var aiRequest: AIRequest?

    private enum ReviewScreen: Identifiable {
        case photo(ScoredPhoto)
        case comparison(groupID: Int)

        var id: String {
            switch self {
            case .photo(let photo): "photo-\(photo.id)"
            case .comparison(let groupID): "group-\(groupID)"
            }
        }
    }

    private enum HomeSheet: String, Identifiable {
        case albums
        case guide
        var id: String { rawValue }
    }

    private struct AIRequest: Identifiable {
        let id = UUID()
        let photos: [ScoredPhoto]
    }

    private let albumColumns = [GridItem(.flexible(), spacing: 12), GridItem(.flexible(), spacing: 12)]
    // Keep review cards readable on iPhone; use two columns only when the available
    // width is large enough (for example, on iPad).
    private let photoColumns = [GridItem(.adaptive(minimum: 320), spacing: 12)]

    var body: some View {
        NavigationStack {
            Group {
                if session.selectedAlbum == nil || session.photos.isEmpty && !session.isAnalyzing {
                    albumHome
                } else {
                    reviewWorkspace
                }
            }
            .background(Color(red: 0.035, green: 0.045, blue: 0.055))
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { activeHomeSheet = .guide } label: {
                        Label("使用说明", systemImage: "info.circle.fill")
                    }
                    .accessibilityHint("查看拾片的筛选和整理方式")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if session.selectedAlbum != nil {
                        Button { session.leaveAlbum() } label: { Label("退出筛选", systemImage: "rectangle.portrait.and.arrow.right") }
                    } else {
                        Button { activeHomeSheet = .albums } label: { Label("选择相册", systemImage: "rectangle.stack") }
                    }
                }
            }
            .sheet(item: $activeHomeSheet) { sheet in
                switch sheet {
                case .albums: albumSheet
                case .guide: guideSheet
                }
            }
            .sheet(item: $aiRequest) { request in
                AIAdviceView(photos: request.photos)
            }
            .fullScreenCover(item: $presentedScreen) { screen in
                switch screen {
                case .photo(let photo):
                    PhotoPreviewView(photo: photo)
                case .comparison(let groupID):
                    GroupComparisonView(groupID: groupID,
                                        photos: session.photos.filter { $0.groupID == groupID })
                }
            }
            .alert("整理完成", isPresented: $showSavedMessage) {
                if library.lastOrganizationReceipt != nil {
                    Button("撤销整理", role: .destructive) {
                        Task {
                            do { session.message = try await library.undoLastOrganization() }
                            catch { session.message = "撤销失败：\(error.localizedDescription)" }
                        }
                    }
                }
                Button("好", role: .cancel) { }
            } message: { Text(session.message ?? "") }
            .confirmationDialog("确认整理这些照片？", isPresented: $showOrganizeConfirmation, titleVisibility: .visible) {
                Button("确认整理", role: .destructive) {
                    guard !isOrganizing else { return }
                    isOrganizing = true
                    Task {
                        await session.saveToAlbums(library: library)
                        isOrganizing = false
                    }
                }
                Button("再检查一下", role: .cancel) { }
            } message: {
                Text(organizeConfirmationText)
            }
            .onChange(of: session.message) { _, value in
                if value != nil { showSavedMessage = true }
            }
            .task {
                if library.authorization == .notDetermined { await library.requestAccess() }
                else { library.loadAlbums() }
            }
        }
        .tint(Color.limeAccent)
    }

    private var albumHome: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("基础分析在 iPhone 上完成")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.limeAccent)
                    Text("拍得多，也能更快挑出\n值得留下的照片。")
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .fixedSize(horizontal: false, vertical: true)
                    Text("选择一个相册，拾片会检查清晰度、曝光和相似照片。拿不准时可主动请求 AI 建议，所有判断都可以手动调整。")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.64))
                        .lineSpacing(4)
                }
                .padding(.top, 20)

                privacyCard

                if library.authorization == .denied || library.authorization == .restricted {
                    ContentUnavailableView("需要照片访问权限", systemImage: "photo.badge.exclamationmark",
                                           description: Text("请在“设置 > 隐私与安全性 > 照片”中允许拣片访问照片库。"))
                } else if library.isLoading {
                    ProgressView("正在读取相册…").frame(maxWidth: .infinity).padding(40)
                } else {
                    HStack {
                        Text("选择待整理的相册").font(.title3.bold())
                        Spacer()
                        Text("\(library.albums.count) 个相册").font(.caption).foregroundStyle(.secondary)
                    }
                    LazyVGrid(columns: albumColumns, spacing: 12) {
                        ForEach(library.albums) { album in
                            Button { Task { await begin(album) } } label: { albumTile(album) }
                                .buttonStyle(.plain)
                        }
                    }
                }
                Text("你的分类选择会自动保存在本机；再次打开同一相册时会重新分析并恢复这些选择。")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.45))
                    .padding(.top, 4)
            }
            .padding(20)
        }
    }

    private var privacyCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "lock.shield.fill").foregroundStyle(Color.limeAccent).font(.title3)
            VStack(alignment: .leading, spacing: 4) {
                Text("基础分析在本机完成").font(.subheadline.bold())
                Text("只有你主动使用 AI 建议时，所选照片的预览图才会发送到 OpenAI。淘汰照片不会被自动删除。")
                    .font(.caption).foregroundStyle(.white.opacity(0.58))
            }
        }
        .padding(16)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18))
    }

    private func albumTile(_ album: AlbumSummary) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ZStack(alignment: .bottomTrailing) {
                AlbumThumbnail(collection: album.collection)
                    .frame(height: 112)
                    .clipShape(RoundedRectangle(cornerRadius: 13))
                Image(systemName: "arrow.up.right")
                    .font(.caption.bold())
                    .padding(8)
                    .background(.black.opacity(0.58), in: Circle())
                    .padding(8)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(album.title).font(.subheadline.bold()).lineLimit(1)
                Text("\(album.count) 张照片").font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(.white.opacity(0.07)))
    }

    private var reviewWorkspace: some View {
        VStack(spacing: 0) {
            if session.isAnalyzing {
                analysisProgress
            } else {
                resultHeader
                filterBar
                ScrollView {
                    LazyVGrid(columns: photoColumns, spacing: 12) {
                        ForEach(session.visiblePhotos()) { photo in photoCard(photo) }
                    }
                    .padding(14)
                }
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    bottomActions
                }
            }
        }
    }

    private var analysisProgress: some View {
        VStack(spacing: 18) {
            HStack {
                Spacer()
                Button("返回相册列表") { session.leaveAlbum() }
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Color.limeAccent)
            }
            .padding(.horizontal, 18).padding(.top, 10)
            Spacer()
            Image(systemName: "sparkles.viewfinder")
                .font(.system(size: 52)).foregroundStyle(Color.limeAccent)
            Text(session.selectedAlbum?.title ?? "正在分析").font(.title2.bold())
            Text(session.statusText).font(.subheadline).foregroundStyle(.secondary)
            ProgressView(value: session.progress)
                .tint(Color.limeAccent)
                .padding(.horizontal, 32)
            Text("分析只使用照片预览，不会修改原片。你可以离开后再回来继续。")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).padding(.horizontal, 42)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var resultHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(session.selectedAlbum?.title ?? "筛选结果").font(.title2.bold())
                Spacer(minLength: 8)
                Button { session.leaveAlbum() } label: {
                    Label("返回相册", systemImage: "arrow.uturn.backward")
                        .font(.caption.weight(.semibold))
                }
                .foregroundStyle(Color.limeAccent)
            }
            HStack(spacing: 8) {
                countPill("保留", count: session.keptCount, color: .limeAccent)
                countPill("待确认", count: session.reviewCount, color: .orange)
                countPill("淘汰", count: session.rejectedCount, color: .pink)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16).padding(.top, 8).padding(.bottom, 12)
    }

    private func countPill(_ title: String, count: Int, color: Color) -> some View {
        HStack(spacing: 5) {
            Text("\(count)").fontWeight(.bold).foregroundStyle(color)
            Text(title).foregroundStyle(.white.opacity(0.7))
        }
        .font(.caption)
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(.white.opacity(0.06), in: Capsule())
    }

    private var filterBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                filterButton("全部", decision: nil)
                ForEach(PhotoDecision.allCases, id: \.self) { decision in
                    filterButton(decision.title, decision: decision)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
        }
    }

    private func filterButton(_ title: String, decision: PhotoDecision?) -> some View {
        let active = session.filter == decision
        return Button { session.filter = decision } label: {
            Text(title).font(.caption.weight(.semibold))
                .padding(.horizontal, 14).padding(.vertical, 8)
                .foregroundStyle(active ? .black : .white.opacity(0.75))
                .background(active ? Color.limeAccent : .white.opacity(0.07), in: Capsule())
        }
    }

    private func photoCard(_ photo: ScoredPhoto) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            ZStack(alignment: .topTrailing) {
                Button { presentedScreen = .photo(photo) } label: {
                    Group {
                        if let thumbnail = photo.thumbnail { Image(uiImage: thumbnail).resizable().scaledToFill() }
                        else { Rectangle().fill(.white.opacity(0.07)).overlay(Image(systemName: "photo")) }
                    }
                    .frame(height: 220)
                    .frame(maxWidth: .infinity)
                    .clipped()
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("查看大图：\(photo.filename)")
                if let groupID = photo.groupID {
                    Text("对比组 #\(groupID) · 第\(photo.groupRank)名")
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 8).padding(.vertical, 5)
                        .background(.black.opacity(0.72), in: Capsule())
                        .padding(8)
                }
                Text("\(photo.score)")
                    .font(.caption.bold()).foregroundStyle(Color.limeAccent)
                    .padding(8).background(.black.opacity(0.72), in: Circle()).padding(8).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text(photo.filename).font(.caption.bold()).lineLimit(1)
                if photo.pairID != nil {
                    Label("RAW + JPG 联动", systemImage: "link")
                        .font(.system(size: 9, weight: .semibold)).foregroundStyle(Color.limeAccent)
                }
                Text(photo.reason).font(.caption2).foregroundStyle(.white.opacity(0.65)).lineLimit(2).frame(height: 30, alignment: .topLeading)
                HStack(spacing: 4) {
                    Text("清晰 \(photo.sharpness)")
                    Text("·")
                    Text("曝光 \(photo.exposure)")
                }.font(.system(size: 9, weight: .medium)).foregroundStyle(.white.opacity(0.48))
                Text("构图 \(photo.compositionScore) · 位置 \(photo.compositionPositionScore) · 平衡 \(photo.compositionBalanceScore) · 边缘 \(photo.compositionEdgeScore)")
                    .font(.system(size: 9, weight: .medium)).foregroundStyle(.white.opacity(0.56))
                    .lineLimit(2).frame(height: 26, alignment: .topLeading)
                HStack(spacing: 5) {
                    ForEach(PhotoDecision.allCases, id: \.self) { decision in
                        Button { session.decide(photo, as: decision) } label: {
                            Text(decision.title).font(.system(size: 10, weight: .bold))
                                .frame(maxWidth: .infinity).padding(.vertical, 8)
                                .foregroundStyle(photo.decision == decision ? .black : .white.opacity(0.75))
                                .background(photo.decision == decision ? decisionColor(decision) : .white.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                if photo.groupID != nil {
                    HStack(spacing: 6) {
                        Button { presentedScreen = .comparison(groupID: photo.groupID!) } label: {
                            Label("查看对比组（\(session.photos.filter { $0.groupID == photo.groupID }.count)）", systemImage: "square.on.square")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundStyle(Color.limeAccent)
                                .frame(maxWidth: .infinity).padding(.vertical, 8)
                                .background(Color.limeAccent.opacity(0.09), in: RoundedRectangle(cornerRadius: 8))
                        }
                        Button { session.setGroupDecision(photo, as: photo.decision) } label: {
                            Label("整组设为「\(photo.decision.title)」", systemImage: "square.stack.3d.up")
                                .font(.system(size: 10, weight: .medium)).foregroundStyle(.white.opacity(0.72))
                                .frame(maxWidth: .infinity).padding(.vertical, 8)
                                .background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                Button { aiRequest = AIRequest(photos: [photo]) } label: {
                    Label("AI 帮我看看", systemImage: "sparkles")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(maxWidth: .infinity).padding(.vertical, 8)
                        .foregroundStyle(Color.limeAccent)
                        .background(Color.limeAccent.opacity(0.09), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            .padding(.horizontal, 9).padding(.bottom, 9)
        }
        .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 15))
        .clipShape(RoundedRectangle(cornerRadius: 15))
        .overlay(RoundedRectangle(cornerRadius: 15).stroke(.white.opacity(0.06)))
    }

    private var bottomActions: some View {
        VStack(spacing: 7) {
            Button { showOrganizeConfirmation = true } label: {
                Label(isOrganizing ? "正在整理…" : "确认筛选并整理相册", systemImage: "checkmark.circle.fill")
                    .font(.subheadline.bold()).frame(maxWidth: .infinity).padding(.vertical, 15)
                    .foregroundStyle(.black).background(Color.limeAccent, in: RoundedRectangle(cornerRadius: 14))
            }
            .disabled(isOrganizing)
            if let album = session.selectedAlbum {
                Text(album.collection.assetCollectionType == .album
                     ? "保留留在“\(album.title)”；待确认／待删除移入“\(album.title)_待确认”和“\(album.title)_待删除”。不会删除图库原片。"
                     : "系统智能相册不能移出照片；待确认／待删除会加入“\(album.title)_待确认”和“\(album.title)_待删除”，原片不会删除。")
                    .font(.system(size: 10)).foregroundStyle(.white.opacity(0.48)).multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal, 16).padding(.top, 9).padding(.bottom, 10)
        .background(.ultraThinMaterial)
    }

    private var organizeConfirmationText: String {
        guard let album = session.selectedAlbum else { return "请先选择相册。" }
        let sourceBehavior = album.collection.assetCollectionType == .album
            ? "待确认和待删除会从来源相册移出，并加入对应结果相册。"
            : "智能相册中的照片不会被移出，会加入对应结果相册。"
        return "保留 \(session.keptCount) 张，待确认 \(session.reviewCount) 张，待删除 \(session.rejectedCount) 张。\n\(sourceBehavior)不会删除系统图库中的照片；整理完成后可立即撤销。"
    }

    private var albumSheet: some View {
        NavigationStack {
            List(library.albums) { album in
                Button {
                    activeHomeSheet = nil
                    Task { await begin(album) }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(album.title).foregroundStyle(.primary)
                            Text("\(album.count) 张照片").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if session.selectedAlbum?.id == album.id { Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.limeAccent) }
                    }.padding(.vertical, 3)
                }
            }
            .navigationTitle("选择相册")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("完成") { activeHomeSheet = nil } } }
        }
        .presentationDetents([.medium, .large])
    }

    private var guideSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    guideRow("1", title: "选择相册", detail: "拾片会在本机读取照片，检查清晰度、曝光和相似照片。")
                    guideRow("2", title: "逐张调整", detail: "把照片设为保留、待确认或淘汰；拿不准时可手动请求 AI 分析单张或对比组。")
                    guideRow("3", title: "确认整理", detail: "待确认和待删除会归入对应系统相册，方便你之后复核。")
                    Label("不会自动删除原片；最终删除由你在系统照片 App 中确认。", systemImage: "lock.shield")
                        .font(.footnote).foregroundStyle(.secondary)
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
                }
                .padding(20)
            }
            .navigationTitle("拾片怎么用")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("完成") { activeHomeSheet = nil } } }
        }
        .presentationDetents([.medium, .large])
    }

    private func guideRow(_ number: String, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number)
                .font(.subheadline.bold()).foregroundStyle(.black)
                .frame(width: 28, height: 28)
                .background(Color.limeAccent, in: Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.subheadline.bold())
                Text(detail).font(.footnote).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func begin(_ album: AlbumSummary) async {
        await session.start(album: album, library: library)
    }

    private func decisionColor(_ decision: PhotoDecision) -> Color {
        switch decision {
        case .keep: Color.limeAccent
        case .review: .orange
        case .reject: .pink
        }
    }
}

private struct GroupComparisonView: View {
    let groupID: Int
    let photos: [ScoredPhoto]
    @Environment(\.dismiss) private var dismiss
    @State private var selectedIndex = 0
    @State private var previewPhoto: ScoredPhoto?
    @State private var showAIAdvice = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("对比组 #\(groupID)").font(.headline.bold())
                    Text("\(photos.count) 张相似照片 · 左右滑动比较")
                        .font(.caption).foregroundStyle(.white.opacity(0.58))
                }
                Spacer()
                Button("AI 帮我选") { showAIAdvice = true }
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Color.limeAccent)
                Button("完成") { dismiss() }
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .background(.white.opacity(0.12), in: Capsule())
            }
            .padding(.horizontal, 16).padding(.vertical, 12)

            if photos.isEmpty {
                ContentUnavailableView("对比组已不存在", systemImage: "photo.on.rectangle.angled")
                    .foregroundStyle(.white)
            } else {
                TabView(selection: $selectedIndex) {
                    ForEach(Array(photos.enumerated()), id: \.element.id) { index, photo in
                        comparisonPage(photo)
                            .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .always))
            }
        }
        .foregroundStyle(.white)
        .background {
            Color(red: 0.035, green: 0.045, blue: 0.055).ignoresSafeArea()
        }
        .preferredColorScheme(.dark)
        .fullScreenCover(item: $previewPhoto) { photo in
            PhotoPreviewView(photo: photo)
        }
        .sheet(isPresented: $showAIAdvice) {
            AIAdviceView(photos: Array(photos.sorted { $0.groupRank < $1.groupRank }.prefix(6)))
        }
    }

    private func comparisonPage(_ photo: ScoredPhoto) -> some View {
        VStack(spacing: 0) {
            Button { previewPhoto = photo } label: {
                Group {
                    if let thumbnail = photo.thumbnail {
                        Image(uiImage: thumbnail).resizable().scaledToFit()
                    } else {
                        ContentUnavailableView("照片预览不可用", systemImage: "photo")
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("查看大图：\(photo.filename)")

            VStack(alignment: .leading, spacing: 9) {
                HStack(alignment: .firstTextBaseline) {
                    Text(photo.filename).font(.subheadline.bold()).lineLimit(1)
                    Spacer(minLength: 8)
                    Text("\(photo.score) 分")
                        .font(.headline.bold()).foregroundStyle(Color.limeAccent)
                }
                HStack(spacing: 8) {
                    Text("第 \(photo.groupRank) 名")
                    Text("·")
                    Text("清晰 \(photo.sharpness)")
                    Text("·")
                    Text("曝光 \(photo.exposure)")
                }
                .font(.caption).foregroundStyle(.white.opacity(0.62))
                Text(photo.reason).font(.caption).foregroundStyle(.white.opacity(0.78))
                Text("构图参考：综合 \(photo.compositionScore) 分（主体位置 \(photo.compositionPositionScore)、画面平衡 \(photo.compositionBalanceScore)、边缘安全 \(photo.compositionEdgeScore)）。三分法参考 \(photo.compositionThirdsScore) 分；水平线倾斜约 \(Int(abs(photo.compositionTilt).rounded()))°。\(photo.compositionNote)。仅作辅助参考，不代表审美结论。")
                    .font(.caption2).foregroundStyle(.white.opacity(0.62))
                Label("当前：\(photo.decision.title)", systemImage: "checkmark.circle")
                    .font(.caption.weight(.semibold)).foregroundStyle(decisionColor(photo.decision))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(.white.opacity(0.06))
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 24)
    }

    private func decisionColor(_ decision: PhotoDecision) -> Color {
        switch decision {
        case .keep: Color.limeAccent
        case .review: .orange
        case .reject: .pink
        }
    }
}

private struct PhotoPreviewView: View {
    let photo: ScoredPhoto
    @Environment(\.dismiss) private var dismiss
    @State private var image: UIImage?
    @State private var requestID: PHImageRequestID?
    @State private var zoomScale: CGFloat = 1
    @State private var baseScale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var baseOffset: CGSize = .zero

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(photo.filename).font(.subheadline.bold()).lineLimit(1)
                    Text("双指放大 · 双击切换缩放")
                        .font(.caption2).foregroundStyle(.white.opacity(0.55))
                }
                Spacer(minLength: 8)
                Button("完成") { dismiss() }
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .background(.white.opacity(0.12), in: Capsule())
            }
            .padding(.horizontal, 16).padding(.vertical, 12)

            GeometryReader { proxy in
                ZStack {
                    Color.black
                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(width: proxy.size.width, height: proxy.size.height)
                            .scaleEffect(zoomScale)
                            .offset(offset)
                            .gesture(
                                DragGesture()
                                    .onChanged { value in
                                        guard zoomScale > 1 else { return }
                                        offset = CGSize(width: baseOffset.width + value.translation.width,
                                                        height: baseOffset.height + value.translation.height)
                                    }
                                    .onEnded { _ in baseOffset = offset }
                            )
                            .simultaneousGesture(
                                MagnificationGesture()
                                    .onChanged { value in zoomScale = min(max(baseScale * value, 1), 5) }
                                    .onEnded { _ in
                                        baseScale = zoomScale
                                        if zoomScale == 1 { offset = .zero; baseOffset = .zero }
                                    }
                            )
                            .onTapGesture(count: 2) {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    zoomScale = zoomScale > 1 ? 1 : 2
                                    baseScale = zoomScale
                                    if zoomScale == 1 { offset = .zero; baseOffset = .zero }
                                }
                            }
                    } else {
                        ProgressView().tint(.white)
                    }
                }
                .frame(width: proxy.size.width, height: proxy.size.height)
                .clipped()
            }
            .background(.black)
        }
        .foregroundStyle(.white)
        .background(Color.black)
        .preferredColorScheme(.dark)
        .onAppear(perform: loadPreview)
        .onDisappear {
            if let requestID { PHImageManager.default().cancelImageRequest(requestID) }
        }
    }

    private func loadPreview() {
        image = photo.thumbnail
        let longestSide = CGFloat(max(photo.asset.pixelWidth, photo.asset.pixelHeight))
        let scale = longestSide > 0 ? min(1, 4096 / longestSide) : 1
        let targetSize = CGSize(width: CGFloat(photo.asset.pixelWidth) * scale,
                                height: CGFloat(photo.asset.pixelHeight) * scale)
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true
        requestID = PHImageManager.default().requestImage(for: photo.asset,
                                                          targetSize: targetSize,
                                                          contentMode: .aspectFit,
                                                          options: options) { result, _ in
            guard let result else { return }
            DispatchQueue.main.async { image = result }
        }
    }
}

private struct AlbumThumbnail: View {
    let collection: PHAssetCollection
    @State private var image: UIImage?

    var body: some View {
        Group {
            if let image { Image(uiImage: image).resizable().scaledToFill() }
            else { Rectangle().fill(.white.opacity(0.08)).overlay(Image(systemName: "photo.on.rectangle.angled").foregroundStyle(.white.opacity(0.45))) }
        }
        .task {
            let assets = PHAsset.fetchAssets(in: collection, options: nil)
            guard let asset = assets.firstObject else { return }
            let options = PHImageRequestOptions()
            options.deliveryMode = .fastFormat
            options.isNetworkAccessAllowed = true
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 320, height: 220), contentMode: .aspectFill, options: options) { result, _ in
                image = result
            }
        }
    }
}

private extension Color {
    static let limeAccent = Color(red: 0.83, green: 0.98, blue: 0.31)
}

private enum AIKeyStore {
    private static let account = "openai-api-key"

    static func load() -> String {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: "jianpian.ai",
                                    kSecAttrAccount as String: account,
                                    kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }

    static func save(_ key: String) -> Bool {
        let identity: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                       kSecAttrService as String: "jianpian.ai",
                                       kSecAttrAccount as String: account]
        SecItemDelete(identity as CFDictionary)
        guard !key.isEmpty else { return true }
        var item = identity
        item[kSecValueData as String] = Data(key.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        return SecItemAdd(item as CFDictionary, nil) == errSecSuccess
    }
}

private struct AIAdviceView: View {
    let photos: [ScoredPhoto]
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var library: PhotoLibraryStore
    @State private var apiKey = AIKeyStore.load()
    @State private var advice = ""
    @State private var error = ""
    @State private var isLoading = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(photos.count == 1 ? "让 AI 看看这张" : "让 AI 比较这 \(photos.count) 张")
                        .font(.title2.bold())
                    Text("仅在你点击下方按钮后，才发送所选照片的压缩预览图到 OpenAI。不会发送 RAW 或整本相册，也不会自动改动筛选结果。调用可能产生 API 费用。")
                        .font(.footnote).foregroundStyle(.secondary)
                    SecureField("OpenAI API Key（sk-…）", text: $apiKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textContentType(.password)
                        .padding(12)
                        .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                    Text("密钥只保存在这台 iPhone 的钥匙串中；ChatGPT 订阅不能代替 API Key。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button {
                        Task { await analyze() }
                    } label: {
                        Label(isLoading ? "正在分析…" : "发送预览图并获取建议", systemImage: "sparkles")
                            .frame(maxWidth: .infinity).padding(14)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isLoading)
                    if isLoading { ProgressView("正在查看画面与比较差异…") }
                    if !error.isEmpty { Text(error).foregroundStyle(.orange).font(.footnote) }
                    if !advice.isEmpty {
                        Text(advice)
                            .font(.body)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(16)
                            .background(.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 14))
                    }
                }
                .padding(20)
            }
            .navigationTitle("AI 选片建议")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("完成") { dismiss() } } }
        }
        .preferredColorScheme(.dark)
    }

    @MainActor
    private func analyze() async {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard AIKeyStore.save(key) else { error = "密钥保存失败，请重试。"; return }
        error = ""
        advice = ""
        isLoading = true
        defer { isLoading = false }
        do {
            var content: [[String: Any]] = [["type": "input_text", "text":
                "你是摄影选片助手。请用中文给出明确但谨慎的筛选建议。单张建议保留/待确认/淘汰；多张指出最值得保留的一张和其他照片的去留。"
                + "结合主体、构图、光线、情绪、清晰度和重复度说明可见证据；不能判断的地方要指出。已有分数只作参考，不要编造细节。建议不代替用户决定。"]]
            for photo in photos.prefix(6) {
                guard let image = await library.thumbnail(for: photo.asset, size: CGSize(width: 1200, height: 1200)),
                      let data = image.jpegData(compressionQuality: 0.78) else {
                    throw NSError(domain: "AIAdvice", code: 1, userInfo: [NSLocalizedDescriptionKey: "无法读取 \(photo.filename) 的预览图"])
                }
                content.append(["type": "input_text", "text": "\(photo.filename)：本地评分 \(photo.score)，当前 \(photo.decision.title)；\(photo.reason)"])
                content.append(["type": "input_image", "image_url": "data:image/jpeg;base64,\(data.base64EncodedString())", "detail": "high"])
            }
            let body: [String: Any] = ["model": "gpt-6-luna", "input": [["role": "user", "content": content]],
                                       "max_output_tokens": 600, "store": false,
                                       "reasoning": ["effort": "none"]]
            var request = URLRequest(url: URL(string: "https://api.openai.com/v1/responses")!)
            request.httpMethod = "POST"
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            request.timeoutInterval = 60
            let (data, response) = try await URLSession.shared.data(for: request)
            let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                let detail = (json["error"] as? [String: Any])?["message"] as? String ?? "请求失败"
                throw NSError(domain: "AIAdvice", code: 2, userInfo: [NSLocalizedDescriptionKey: detail])
            }
            let output = json["output"] as? [[String: Any]] ?? []
            let text = output.flatMap { $0["content"] as? [[String: Any]] ?? [] }
                .compactMap { $0["text"] as? String }.joined(separator: "\n")
            guard !text.isEmpty else { throw NSError(domain: "AIAdvice", code: 3, userInfo: [NSLocalizedDescriptionKey: "模型没有返回可显示的建议"])}
            advice = text
        } catch {
            self.error = "暂时无法获取建议：\(error.localizedDescription)"
        }
    }
}
