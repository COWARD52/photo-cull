import SwiftUI

@main
struct PhotoCullApp: App {
    @StateObject private var library = PhotoLibraryStore()
    @StateObject private var session = CullingSession()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(library)
                .environmentObject(session)
                .preferredColorScheme(.dark)
        }
    }
}
