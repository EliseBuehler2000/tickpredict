import XCTest

/// End-to-end flows on real development photos (Simulator only: the photo path is on this Mac).
@MainActor
final class ScanFlowTests: XCTestCase {
    static let photos = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("dataset/Penn Data/433 tick photos")
    static let sample = photos.appendingPathComponent("279926_e9cb902347094e2c791e9df4505ebc4a.jpg").path

    var app: XCUIApplication!

    override func setUp() async throws {
        continueAfterFailure = false
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.sample), "Development photos not available")
        app = XCUIApplication()
        app.launchArguments += ["-resetData", "YES"]
    }

    func launchWithScan() {
        app.launchArguments += ["-demoImage", Self.sample]
        app.launch()
        XCTAssertTrue(app.staticTexts["What this means"].waitForExistence(timeout: 60), "Result never appeared")
    }

    /// The species card combines its text into one accessibility element.
    var speciesCard: XCUIElement { app.descendants(matching: .any).matching(identifier: "speciesCard").firstMatch }

    func saveScan(place: String? = nil) {
        app.navigationBars.buttons["Save"].tap()
        XCTAssertTrue(app.navigationBars["Save scan"].waitForExistence(timeout: 5))
        if let place {
            let field = app.textFields["e.g. Backyard, hiking trail"]
            field.tap()
            field.typeText(place)
        }
        app.buttons["Back of knee"].tap()
        app.navigationBars["Save scan"].buttons["Save"].tap()
    }

    func testSaveShowsNotesAndAppearsInHistory() {
        launchWithScan()
        saveScan(place: "Trail")
        XCTAssertTrue(app.staticTexts["Your notes"].waitForExistence(timeout: 5), "Saved notes card missing")
        XCTAssertEqual(app.state, .runningForeground)
        app.tabBars.buttons["History"].tap()
        XCTAssertTrue(app.staticTexts["Blacklegged tick · Nymph · Semi-engorged"].waitForExistence(timeout: 5))
    }

    /// A fully engorged blacklegged nymph the species model misnames: the Lyme card must still warn.
    func testEngorgedTickGetsLymeWarning() {
        app.launchArguments += ["-demoImage", Self.photos.appendingPathComponent("289422_0e5badda6f8eb1b4a048095707a239f2.jpg").path]
        app.launch()
        XCTAssertTrue(speciesCard.waitForExistence(timeout: 60))
        let card = app.staticTexts["Higher Lyme risk"]
        for _ in 0..<8 where !card.isHittable { app.swipeUp() }
        XCTAssertTrue(card.exists)
        XCTAssertFalse(app.staticTexts["Doesn't spread Lyme"].exists)
    }

    /// Species is stored with the scan and survives a relaunch.
    func testSavedScanKeepsSpeciesAfterRelaunch() {
        launchWithScan()
        saveScan()
        XCTAssertTrue(app.staticTexts["Your notes"].waitForExistence(timeout: 5))
        app.terminate()
        app.launchArguments = ["-tab", "history"]
        app.launch()
        let row = app.staticTexts["Blacklegged tick · Nymph · Semi-engorged"]
        XCTAssertTrue(row.waitForExistence(timeout: 10))
        row.tap()
        XCTAssertTrue(speciesCard.waitForExistence(timeout: 10), "Species missing after relaunch")
        XCTAssertTrue(speciesCard.label.contains("Ixodes scapularis"))
    }

    func testSaveButtonAtBottomOfResult() {
        launchWithScan()
        let button = app.buttons["Save to history"]
        for _ in 0..<10 where !button.isHittable { app.swipeUp() }
        button.tap()
        app.navigationBars["Save scan"].buttons["Save"].tap()
        XCTAssertTrue(app.staticTexts["Your notes"].waitForExistence(timeout: 5))
    }

    func testEditThenDeleteSavedScan() {
        launchWithScan()
        saveScan()
        app.navigationBars.buttons["Scan options"].tap()
        app.buttons["Edit details"].tap()
        XCTAssertTrue(app.navigationBars["Edit scan"].waitForExistence(timeout: 5))
        app.buttons["Arm"].tap()
        app.navigationBars["Edit scan"].buttons["Done"].tap()
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS 'Arm'")).firstMatch.waitForExistence(timeout: 5),
                      "Edited bite spot not shown")
        app.navigationBars.buttons["Scan options"].tap()
        app.buttons["Delete scan"].tap()
        app.buttons["Delete scan"].firstMatch.tap()
        XCTAssertTrue(app.buttons["Open camera"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.state, .runningForeground)
    }

    func testDetailsToggleAndFindCare() {
        launchWithScan()
        let details = app.buttons["Details"]
        for _ in 0..<10 where !details.isHittable { app.swipeUp() }
        details.tap()
        XCTAssertTrue(app.staticTexts["Life stage scores".uppercased()].waitForExistence(timeout: 3))
        app.swipeDown(); app.swipeDown()
        let care = app.buttons["Find care nearby"].firstMatch
        for _ in 0..<10 where !care.isHittable { app.swipeUp() }
        care.tap()
        XCTAssertTrue(app.navigationBars["Find care"].waitForExistence(timeout: 5))
    }

    /// Both Find care buttons on a result (Lyme card and next steps) open a working
    /// screen that survives the location prompt and the search.
    func testFindCareFromResultDoesNotCrash() {
        launchWithScan()
        for index in 0..<2 {
            let care = app.buttons.matching(identifier: "Find care nearby").element(boundBy: index)
            for _ in 0..<10 where !care.isHittable { app.swipeUp() }
            care.tap()
            XCTAssertTrue(app.navigationBars["Find care"].waitForExistence(timeout: 5))
            let allow = XCUIApplication(bundleIdentifier: "com.apple.springboard").buttons["Allow While Using App"]
            if allow.waitForExistence(timeout: 5) { allow.tap() }
            sleep(12)
            XCTAssertEqual(app.state, .runningForeground, "Crashed on Find care (button \(index))")
            app.navigationBars.buttons.element(boundBy: 0).tap()
            XCTAssertTrue(app.navigationBars["Result"].waitForExistence(timeout: 5))
        }
    }

    func testFindCareAllowOnce() { findCareFirstTime(answer: "Allow Once") }
    func testFindCareDontAllow() { findCareFirstTime(answer: "Don’t Allow") }

    /// First visit to Find care, answering the location prompt with `answer`.
    /// Run after `xcrun simctl privacy <device> reset location com.ericbuehler.TickPredict`.
    func findCareFirstTime(answer: String) {
        launchWithScan()
        let care = app.buttons.matching(identifier: "Find care nearby").firstMatch
        for _ in 0..<10 where !care.isHittable { app.swipeUp() }
        care.tap()
        XCTAssertTrue(app.navigationBars["Find care"].waitForExistence(timeout: 5))
        let button = XCUIApplication(bundleIdentifier: "com.apple.springboard").buttons[answer]
        if button.waitForExistence(timeout: 8) { button.tap() } else { XCTFail("No location prompt") }
        sleep(12)
        XCTAssertEqual(app.state, .runningForeground, "Crashed after \(answer)")
    }

    func testDeletingInHistoryClosesOpenScan() {
        launchWithScan()
        saveScan()
        app.tabBars.buttons["History"].tap()
        let row = app.staticTexts["Blacklegged tick · Nymph · Semi-engorged"]
        XCTAssertTrue(row.waitForExistence(timeout: 5))
        row.swipeLeft()
        app.buttons["Delete"].tap()
        app.tabBars.buttons["Scan"].tap()
        // The fresh result is still on screen, now unsaved again.
        XCTAssertTrue(app.navigationBars.buttons["Save"].waitForExistence(timeout: 5), "Result didn't return to unsaved")
        XCTAssertFalse(app.staticTexts["Your notes"].exists)
        XCTAssertEqual(app.state, .runningForeground)
    }

    func testDeletingInHistoryClosesSavedScanOpenedFromHome() {
        launchWithScan()
        saveScan()
        app.navigationBars.buttons.element(boundBy: 0).tap()
        let recent = app.staticTexts["Blacklegged tick · Nymph · Semi-engorged"]
        XCTAssertTrue(recent.waitForExistence(timeout: 5))
        recent.tap()
        XCTAssertTrue(app.staticTexts["Your notes"].waitForExistence(timeout: 5))
        app.tabBars.buttons["History"].tap()
        let row = app.cells.staticTexts["Blacklegged tick · Nymph · Semi-engorged"]
        XCTAssertTrue(row.waitForExistence(timeout: 5))
        row.swipeLeft()
        app.buttons["Delete"].tap()
        app.tabBars.buttons["Scan"].tap()
        XCTAssertTrue(app.buttons["Open camera"].waitForExistence(timeout: 5), "Deleted scan still open")
        XCTAssertEqual(app.state, .runningForeground)
    }

    func testCancelWithChangesAsksBeforeDiscarding() {
        launchWithScan()
        app.navigationBars.buttons["Save"].tap()
        app.buttons["Arm"].tap()
        app.navigationBars["Save scan"].buttons["Cancel"].tap()
        XCTAssertTrue(app.buttons["Discard changes"].waitForExistence(timeout: 3))
        app.buttons["Discard changes"].tap()
        XCTAssertTrue(app.buttons["Save to history"].waitForExistence(timeout: 5) || app.navigationBars.buttons["Save"].exists)
    }

    func testBackOutDuringAnalysisThenReopen() {
        app.launchArguments += ["-demoImage", Self.sample]
        app.launch()
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(app.buttons["Open camera"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.state, .runningForeground)
    }
}

/// Runs on a real iPhone too: needs no development photos.
@MainActor
final class FindCareDeviceTests: XCTestCase {
    func testFindCareSurvivesLocationAndSearch() {
        let app = XCUIApplication()
        app.launchArguments += ["-findCare", "YES"]
        app.launch()
        XCTAssertTrue(app.navigationBars["Find care"].waitForExistence(timeout: 15))
        let allow = XCUIApplication(bundleIdentifier: "com.apple.springboard").buttons["Allow While Using App"]
        if allow.waitForExistence(timeout: 5) { allow.tap() }
        for kind in ["Doctors", "Hospitals", "Urgent care"] {
            sleep(6)
            XCTAssertEqual(app.state, .runningForeground, "Crashed before switching to \(kind)")
            app.buttons[kind].tap()
        }
        sleep(8)
        XCTAssertEqual(app.state, .runningForeground)
    }

    /// Needs a photo copied to the app's Documents as sample.jpg (see ios/README.md).
    func testFindCareFromResult() {
        let app = XCUIApplication()
        app.launchArguments += ["-demoImage", "sample.jpg"]
        app.launch()
        let care = app.buttons.matching(identifier: "Find care nearby").firstMatch
        XCTAssertTrue(app.staticTexts["What this means"].waitForExistence(timeout: 60), "Result never appeared")
        for _ in 0..<10 where !care.isHittable { app.swipeUp() }
        care.tap()
        XCTAssertTrue(app.navigationBars["Find care"].waitForExistence(timeout: 10))
        sleep(12)
        XCTAssertEqual(app.state, .runningForeground)
    }
}
