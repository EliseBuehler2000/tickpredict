package com.ericbuehler.tickpredict.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.FloatBuffer

/** One task's answer: the top label plus every class score. Scores are the models' uncalibrated softmax outputs. */
@Serializable
data class TaskPrediction(val label: String, val scores: Map<String, Double>) {
    constructor(scores: Map<String, Double>) : this(scores.maxByOrNull { it.value }?.key ?: "", scores)

    val confidence: Double get() = scores[label] ?: 0.0

    /** Classes ordered from most to least likely. */
    val ranked: List<Pair<String, Double>> get() = scores.entries.sortedByDescending { it.value }.map { it.key to it.value }
}

/** Species is optional as on iOS, where scans saved with v1.0 have none. */
@Serializable
data class TickPrediction(
    val species: TaskPrediction? = null,
    val lifeStage: TaskPrediction,
    val engorgement: TaskPrediction,
    val modelRelease: String,
)

class TickClassifierException(message: String) : Exception(message)

/**
 * Runs the bundled models on a photo: the v2 multi-head model for species and
 * life stage, and the v1 engorgement model, which scored better than v2's head.
 */
class TickClassifier(multiheadPath: File, engorgementPath: File) : AutoCloseable {
    private class Loaded(val session: OrtSession, val spec: PreprocessingSpec, val metadata: Map<String, String>, val release: String)

    private val environment = OrtEnvironment.getEnvironment()
    private val multihead: Loaded
    private val heads: Map<String, List<String>>
    private val engorgement: Loaded
    private val engorgementClasses: List<String>

    init {
        multihead = load(multiheadPath, "multihead")
        engorgement = load(engorgementPath, "engorgement_level")
        val headsJSON = multihead.metadata["heads"] ?: throw TickClassifierException("Missing heads")
        val classesJSON = engorgement.metadata["classes"] ?: throw TickClassifierException("Missing classes")
        heads = json.decodeFromString(headsJSON)
        engorgementClasses = json.decodeFromString(classesJSON)
    }

    private fun load(path: File, task: String): Loaded {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(8))
        }
        val session = environment.createSession(path.absolutePath, options)
        val metadata = session.metadata.customMetadata
        val specJSON = metadata["preprocessing"]
        val release = metadata["release"]
        if (metadata["task"] != task || specJSON == null || release == null) {
            session.close()
            throw TickClassifierException("Missing metadata in ${path.name}")
        }
        return Loaded(session, json.decodeFromString(specJSON), metadata, release)
    }

    /** Runs both models on encoded image data (JPEG, HEIC, PNG…). */
    fun predict(imageData: ByteArray): TickPrediction = predict(ImageDecoding.decode(imageData))

    fun predict(image: RGBImage): TickPrediction {
        val (first, second) = preparedInputs(image)
        val output = run(multihead, first)
        val engorgementScores = run(engorgement, second).getValue("probabilities")
        if (engorgementScores.size != engorgementClasses.size) throw TickClassifierException("Unexpected engorgement output")
        return TickPrediction(
            species = head("species", output),
            lifeStage = head("life_stage", output),
            engorgement = TaskPrediction(engorgementClasses.zip(engorgementScores.map { it.toDouble() }).toMap()),
            modelRelease = multihead.release,
        )
    }

    /** The prepared model inputs, exposed for parity checks. Both models currently share one recipe, so the photo is prepared once. */
    fun preparedInputs(image: RGBImage): Pair<RGBImage, RGBImage> {
        val first = ImagePreparation.prepare(image, multihead.spec)
        val second = if (engorgement.spec == multihead.spec) first else ImagePreparation.prepare(image, engorgement.spec)
        return first to second
    }

    /** 0-255 RGB as float32 NCHW; normalization is inside the model. */
    private fun run(loaded: Loaded, prepared: RGBImage): Map<String, FloatArray> {
        val plane = prepared.width * prepared.height
        val buffer = FloatBuffer.allocate(3 * plane)
        val array = buffer.array()
        val pixels = prepared.pixels
        for (i in 0 until plane) {
            array[i] = (pixels[i * 3].toInt() and 0xFF).toFloat()
            array[plane + i] = (pixels[i * 3 + 1].toInt() and 0xFF).toFloat()
            array[2 * plane + i] = (pixels[i * 3 + 2].toInt() and 0xFF).toFloat()
        }
        val shape = longArrayOf(1, 3, prepared.height.toLong(), prepared.width.toLong())
        OnnxTensor.createTensor(environment, buffer, shape).use { tensor ->
            loaded.session.run(mapOf("image" to tensor)).use { result ->
                return result.associate { (name, value) ->
                    @Suppress("UNCHECKED_CAST")
                    name to (value.value as Array<FloatArray>)[0]
                }
            }
        }
    }

    private fun head(name: String, output: Map<String, FloatArray>): TaskPrediction {
        val classes = heads[name]
        val scores = output[name]
        if (classes == null || scores == null || scores.size != classes.size) throw TickClassifierException("Unexpected output $name")
        return TaskPrediction(classes.zip(scores.map { it.toDouble() }).toMap())
    }

    override fun close() {
        multihead.session.close()
        engorgement.session.close()
    }

    private companion object {
        val json = Json { ignoreUnknownKeys = true }
    }
}
