package com.ericbuehler.tickpredict

import android.content.Context
import android.content.Intent
import com.ericbuehler.tickpredict.model.BodyArea
import com.ericbuehler.tickpredict.model.Scan
import com.ericbuehler.tickpredict.ui.AppTab
import com.ericbuehler.tickpredict.ui.Navigator
import com.ericbuehler.tickpredict.ui.Route
import java.io.File
import java.util.UUID

/**
 * Debug-only launch extras for reviewing screens on an emulator, e.g.
 * `adb shell am start -n com.ericbuehler.tickpredict/.MainActivity --es demoImage sample.jpg`:
 *   --es demoImage photo.jpg        open a new scan of that photo (relative: in the app's external files)
 *   --es tab history|guide          start on a tab
 *   --ez findCare true              push Find care on the Scan tab
 *   --ez camera true                open the camera
 *   --ez resetData true             delete all saved scans first
 *   --es seedScans a.jpg,b.jpg      save scans of those photos (if history is empty)
 */
class DebugLaunch(
    private val demoImage: String?,
    private val tab: String?,
    private val findCare: Boolean,
    private val camera: Boolean,
    private val resetData: Boolean,
    private val seedScans: String?,
) {
    suspend fun apply(context: Context, navigator: Navigator) {
        val store = context.app.store
        if (resetData) store.deleteAll()
        seedScans?.let { seed(context, it) }
        demoImage?.let { resolve(context, it) }?.takeIf { it.exists() }?.let { navigator.open(PendingPhoto(it.readBytes())) }
        when (tab) {
            "history" -> navigator.tab = AppTab.HISTORY
            "guide" -> navigator.tab = AppTab.GUIDE
        }
        if (findCare) navigator.stacks.getValue(AppTab.SCAN).add(Route.FindCare())
        if (camera) navigator.showCamera = true
    }

    private suspend fun seed(context: Context, list: String) {
        val store = context.app.store
        if (store.scans.value.isNotEmpty()) return
        val details = listOf(
            Triple(BodyArea.FOOT, "Backyard", "Probably from mowing on Saturday") to 0,
            Triple(BodyArea.KNEE, "Trail", "") to 6,
            Triple(BodyArea.TORSO, "Campsite", "Saw it two days after the trip") to 21,
            Triple(BodyArea.ARM, "Garden", "") to 40,
        )
        for ((index, path) in list.split(",").withIndex()) {
            val file = resolve(context, path).takeIf { it.exists() } ?: continue
            val data = file.readBytes()
            val prediction = try {
                context.app.classifier.classify(PendingPhoto(data))
            } catch (_: Exception) {
                continue
            }
            val (info, daysAgo) = details[index % details.size]
            val date = System.currentTimeMillis() - daysAgo * 86_400_000L
            store.insert(Scan(UUID.randomUUID().toString(), date, prediction, date, info.first, info.second, info.third), data)
        }
    }

    /** Relative paths are in the app's external files folder, for photos pushed with adb. */
    private fun resolve(context: Context, path: String): File =
        if (path.startsWith("/")) File(path) else File(context.getExternalFilesDir(null), path)

    companion object {
        fun from(intent: Intent?): DebugLaunch? {
            val extras = intent?.extras ?: return null
            return DebugLaunch(
                demoImage = extras.getString("demoImage"),
                tab = extras.getString("tab"),
                findCare = extras.getBoolean("findCare"),
                camera = extras.getBoolean("camera"),
                resetData = extras.getBoolean("resetData"),
                seedScans = extras.getString("seedScans"),
            )
        }
    }
}
