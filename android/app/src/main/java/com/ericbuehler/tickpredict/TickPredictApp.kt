package com.ericbuehler.tickpredict

import android.app.Application
import android.content.Context
import com.ericbuehler.tickpredict.inference.TickClassifier
import com.ericbuehler.tickpredict.inference.TickPrediction
import com.ericbuehler.tickpredict.model.ScanStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.withContext
import org.osmdroid.config.Configuration
import java.io.File
import java.io.IOException
import java.util.UUID

class TickPredictApp : Application() {
    lateinit var classifier: ClassifierService
        private set
    lateinit var store: ScanStore
        private set

    override fun onCreate() {
        super.onCreate()
        classifier = ClassifierService(this)
        store = ScanStore(File(filesDir, "scans"))
        Configuration.getInstance().apply {
            userAgentValue = "TickPredict/${BuildConfig.VERSION_NAME} (Android; $packageName)"
            osmdroidBasePath = File(cacheDir, "osmdroid")
            osmdroidTileCache = File(cacheDir, "osmdroid/tiles")
        }
    }
}

val Context.app: TickPredictApp get() = applicationContext as TickPredictApp

/** A photo waiting to be scanned. Identity is its id. */
class PendingPhoto(val data: ByteArray) {
    val id: String = UUID.randomUUID().toString()
    val takenAt: Long = minOf(PhotoLoader.captureDate(data) ?: System.currentTimeMillis(), System.currentTimeMillis())
}

enum class ScanError { MODELS_UNAVAILABLE, UNREADABLE_PHOTO }

class ScanException(val error: ScanError) : Exception(error.name)

/**
 * Loads the models once in the background and runs scans. Each photo is
 * classified once; returning to its screen reuses the same result.
 */
class ClassifierService(private val context: Context) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var loading: Deferred<TickClassifier> = load()
    private val runs = mutableMapOf<String, Deferred<TickPrediction>>()

    private fun load() = scope.async(Dispatchers.IO) {
        TickClassifier(modelFile("TickPredictV2.onnx"), modelFile("EngorgementLevel.onnx"))
    }

    /**
     * Loading from a file path keeps the models out of the Java heap, so copy each one
     * out of the APK once per install or update.
     */
    private fun modelFile(name: String): File {
        val installed = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        val root = File(context.noBackupFilesDir, "models")
        val directory = File(root, installed.toString())
        val file = File(directory, name)
        if (!file.exists()) {
            root.listFiles()?.filter { it != directory }?.forEach { it.deleteRecursively() }
            directory.mkdirs()
            val partial = File(directory, "$name.partial")
            context.assets.open("models/$name").use { input -> partial.outputStream().use { input.copyTo(it) } }
            if (!partial.renameTo(file)) throw IOException("Couldn't copy $name")
        }
        return file
    }

    suspend fun classify(photo: PendingPhoto): TickPrediction = withContext(Dispatchers.Main) {
        val run = runs[photo.id] ?: start(photo)
        try {
            run.await()
        } catch (error: Exception) {
            runs.remove(photo.id)
            if ((error as? ScanException)?.error == ScanError.MODELS_UNAVAILABLE) loading = load()
            throw error
        }
    }

    private fun start(photo: PendingPhoto): Deferred<TickPrediction> {
        val loader = loading
        val run = scope.async {
            val classifier = try {
                loader.await()
            } catch (_: Exception) {
                throw ScanException(ScanError.MODELS_UNAVAILABLE)
            }
            try {
                classifier.predict(photo.data)
            } catch (error: CancellationException) {
                throw error
            } catch (_: Throwable) {
                // Includes running out of memory on a very large photo.
                throw ScanException(ScanError.UNREADABLE_PHOTO)
            }
        }
        runs[photo.id] = run
        return run
    }
}
