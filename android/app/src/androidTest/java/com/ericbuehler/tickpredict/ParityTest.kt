package com.ericbuehler.tickpredict

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.ericbuehler.tickpredict.inference.ImageDecoding
import com.ericbuehler.tickpredict.inference.TickClassifier
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/**
 * Checks the Android pipeline (decode, EXIF, Lanczos letterbox, ONNX Runtime) against
 * Python on the development photos, like ios/Tools/ParityCheck. Fixtures come from
 * scripts/ios_parity_fixtures.py; push them first (see android/README.md):
 *   adb push FIXTURES /data/local/tmp/parity
 *   adb shell run-as com.ericbuehler.tickpredict cp -r /data/local/tmp/parity files/
 */
@RunWith(AndroidJUnit4::class)
class ParityTest {
    @Test
    fun matchesPython() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "parity")
        val manifest = File(directory, "fixtures.json")
        assumeTrue("No fixtures at $directory", manifest.exists())

        fun model(name: String) = File(context.cacheDir, name).apply {
            if (!exists()) context.assets.open("models/$name").use { input -> outputStream().use { input.copyTo(it) } }
        }
        val classifier = TickClassifier(model("TickPredictV2.onnx"), model("EngorgementLevel.onnx"))
        val entries = Json.parseToJsonElement(manifest.readText()).jsonObject.getValue("entries").jsonArray.map { it.jsonObject }

        var exactInputs = 0
        var worstPixel = 0
        val heads = listOf("species", "life_stage", "engorgement_level")
        val agree = heads.associateWith { 0 }.toMutableMap()
        val correct = heads.associateWith { 0 }.toMutableMap()
        val maxDifference = heads.associateWith { 0.0 }.toMutableMap()
        var compared = 0
        var slowest = 0L
        for (entry in entries) {
            val id = entry.getValue("id").jsonPrimitive.content
            val image = File(directory, File(entry.getValue("image").jsonPrimitive.content).name)
            val expected = File(directory, "${id}_multihead.rgb").readBytes()
            val decoded = ImageDecoding.decode(image.readBytes())
            val prepared = classifier.preparedInputs(decoded).first.pixels
            var worst = 0
            for (i in prepared.indices) worst = max(worst, abs((prepared[i].toInt() and 0xFF) - (expected[i].toInt() and 0xFF)))
            if (worst == 0) exactInputs++
            worstPixel = max(worstPixel, worst)

            val reference = entry["reference"]?.jsonObject ?: continue
            val truth = entry.getValue("truth").jsonObject
            val started = System.nanoTime()
            val prediction = classifier.predict(decoded)
            slowest = max(slowest, (System.nanoTime() - started) / 1_000_000)
            compared++
            val ours = mapOf("species" to prediction.species!!, "life_stage" to prediction.lifeStage, "engorgement_level" to prediction.engorgement)
            for (head in heads) {
                val scores = reference.getValue(head).jsonObject.mapValues { it.value.jsonPrimitive.double }
                val task = ours.getValue(head)
                if (task.label == scores.maxBy { it.value }.key) agree[head] = agree.getValue(head) + 1
                if (task.label == truth[head]?.jsonPrimitive?.contentOrNull) correct[head] = correct.getValue(head) + 1
                val difference = scores.maxOf { (label, score) -> abs((task.scores[label] ?: 0.0) - score) }
                maxDifference[head] = max(maxDifference.getValue(head), difference)
            }
        }
        val summary = buildString {
            appendLine("Prepared inputs: $exactInputs/${entries.size} identical to Python, worst pixel difference $worstPixel")
            for (head in heads) {
                appendLine("$head: ${agree[head]}/$compared agree with PyTorch, ${correct[head]} correct, max |dp| ${"%.4f".format(maxDifference[head])}")
            }
            append("Slowest scan: $slowest ms")
        }
        Log.i("ParityTest", summary)
        File(context.filesDir, "parity-result.txt").writeText(summary)
        for (head in heads) assertTrue(summary, agree.getValue(head) >= compared - 2)
    }
}
