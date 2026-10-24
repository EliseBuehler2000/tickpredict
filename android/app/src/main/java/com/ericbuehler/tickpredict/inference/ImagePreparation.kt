package com.ericbuehler.tickpredict.inference

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.util.stream.IntStream
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** The preprocessing recipe recorded in each ONNX model's metadata. */
@Serializable
data class PreprocessingSpec(
    @SerialName("image_size") val imageSize: Int,
    @SerialName("bottom_crop_fraction") val bottomCropFraction: Double,
    @SerialName("pad_color") val padColor: List<Int>,
)

class ImagePreparationException(message: String) : Exception(message)

/** Tightly packed 8-bit RGB pixels, row major. */
class RGBImage(val width: Int, val height: Int, val pixels: ByteArray) {
    init {
        require(pixels.size == width * height * 3)
    }

    companion object {
        fun filled(width: Int, height: Int, fill: List<Int>): RGBImage {
            val pixels = ByteArray(width * height * 3)
            for (i in pixels.indices step 3) {
                pixels[i] = fill[0].toByte(); pixels[i + 1] = fill[1].toByte(); pixels[i + 2] = fill[2].toByte()
            }
            return RGBImage(width, height, pixels)
        }
    }
}

/**
 * Reproduces the Python training pipeline (Pillow 13) exactly:
 * EXIF transpose -> RGB -> remove bottom strip -> `ImageOps.pad` with Lanczos.
 * A port of ios/TickPredict/Inference/ImagePreparation.swift. Decoding lives in
 * ImageDecoding.kt; everything here is plain Kotlin so it runs in JVM tests too.
 */
object ImagePreparation {
    /** Full pipeline for one model's input. */
    fun prepare(image: RGBImage, spec: PreprocessingSpec): RGBImage {
        if (min(image.width, image.height) < 2) throw ImagePreparationException("Image too small")
        val keptRows = pythonRound(image.height * (1 - spec.bottomCropFraction)).toInt()
        return pad(image, max(keptRows, 1), spec.imageSize, spec.padColor)
    }

    /** Letterboxes the top `rows` rows of `image` (the rest is the removed bottom strip). */
    fun pad(image: RGBImage, rows: Int, size: Int, color: List<Int>): RGBImage {
        var width = size
        var height = size
        val imageRatio = image.width.toDouble() / rows
        if (imageRatio != 1.0) {
            if (imageRatio > 1) {
                height = max(1, pythonRound(rows.toDouble() / image.width * size).toInt())
            } else {
                width = max(1, pythonRound(image.width.toDouble() / rows * size).toInt())
            }
        }
        val resized = LanczosResampler.resize(image, rows, width, height)
        if (width == size && height == size) return resized
        val out = RGBImage.filled(size, size, color)
        val x0 = if (width != size) pythonRound((size - width) * 0.5).toInt() else 0
        val y0 = if (width != size) 0 else pythonRound((size - height) * 0.5).toInt()
        for (y in 0 until height) {
            System.arraycopy(resized.pixels, y * width * 3, out.pixels, ((y + y0) * size + x0) * 3, width * 3)
        }
        return out
    }

    /** Python 3 `round`: halves go to the even neighbour. */
    fun pythonRound(value: Double): Double = Math.rint(value)
}

/** Runs `body` over `count` items split into contiguous chunks on all cores. */
fun parallelChunks(count: Int, body: (IntRange) -> Unit) {
    val chunks = min(count, Runtime.getRuntime().availableProcessors() * 4)
    if (chunks <= 1) {
        body(0 until count)
        return
    }
    // Each chunk writes a disjoint range of rows, so sharing `body` across threads is safe.
    IntStream.range(0, chunks).parallel().forEach { chunk ->
        body(count * chunk / chunks until count * (chunk + 1) / chunks)
    }
}

/**
 * A port of Pillow's 8-bit separable resampler (`libImaging/Resample.c`) with
 * the Lanczos (a = 3) filter, including its fixed-point rounding.
 */
object LanczosResampler {
    private const val PRECISION_BITS = 32 - 8 - 2
    private const val SUPPORT = 3.0

    private fun sinc(x: Double): Double {
        if (x == 0.0) return 1.0
        val px = x * PI
        return sin(px) / px
    }

    private fun lanczos(x: Double): Double = if (-3.0 <= x && x < 3.0) sinc(x) * sinc(x / 3) else 0.0

    private class Coefficients(val kernelSize: Int, val starts: IntArray, val counts: IntArray, val weights: IntArray)

    private fun coefficients(inSize: Int, outSize: Int): Coefficients {
        val scale = inSize.toDouble() / outSize
        val filterScale = max(scale, 1.0)
        val reach = SUPPORT * filterScale
        val kernelSize = ceil(reach).toInt() * 2 + 1
        val starts = IntArray(outSize)
        val counts = IntArray(outSize)
        val weights = IntArray(outSize * kernelSize)
        val raw = DoubleArray(kernelSize)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val inverse = 1 / filterScale
            val xmin = max((center - reach + 0.5).toInt(), 0)
            val xmax = min((center + reach + 0.5).toInt(), inSize) - xmin
            var total = 0.0
            for (x in 0 until xmax) {
                val w = lanczos((x + xmin - center + 0.5) * inverse)
                raw[x] = w
                total += w
            }
            for (x in 0 until xmax) {
                val w = if (total != 0.0) raw[x] / total else raw[x]
                val scaled = w * (1 shl PRECISION_BITS)
                weights[xx * kernelSize + x] = (if (w < 0) -0.5 + scaled else 0.5 + scaled).toInt()
            }
            starts[xx] = xmin
            counts[xx] = xmax
        }
        return Coefficients(kernelSize, starts, counts, weights)
    }

    private fun clip8(value: Int): Byte {
        if (value >= 1 shl (PRECISION_BITS + 8)) return 255.toByte()
        if (value <= 0) return 0
        return (value shr PRECISION_BITS).toByte()
    }

    /** Resizes the top `rows` rows of `image`. */
    fun resize(image: RGBImage, rows: Int, width: Int, height: Int): RGBImage {
        var current = if (width != image.width) {
            horizontal(image, rows, width)
        } else {
            RGBImage(image.width, rows, image.pixels.copyOfRange(0, image.width * rows * 3))
        }
        if (height != rows) current = vertical(current, height)
        return current
    }

    private fun horizontal(image: RGBImage, rows: Int, width: Int): RGBImage {
        val c = coefficients(image.width, width)
        val half = 1 shl (PRECISION_BITS - 1)
        val src = image.pixels
        val k = c.weights
        val out = ByteArray(width * rows * 3)
        parallelChunks(rows) { chunk ->
            for (y in chunk) {
                val row = y * image.width * 3
                for (xx in 0 until width) {
                    val start = c.starts[xx]
                    val count = c.counts[xx]
                    val kk = xx * c.kernelSize
                    var r = half
                    var g = half
                    var b = half
                    for (x in 0 until count) {
                        val p = row + (start + x) * 3
                        val w = k[kk + x]
                        r += (src[p].toInt() and 0xFF) * w
                        g += (src[p + 1].toInt() and 0xFF) * w
                        b += (src[p + 2].toInt() and 0xFF) * w
                    }
                    val o = (y * width + xx) * 3
                    out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b)
                }
            }
        }
        return RGBImage(width, rows, out)
    }

    private fun vertical(image: RGBImage, height: Int): RGBImage {
        val c = coefficients(image.height, height)
        val half = 1 shl (PRECISION_BITS - 1)
        val stride = image.width * 3
        val src = image.pixels
        val k = c.weights
        val out = ByteArray(image.width * height * 3)
        parallelChunks(height) { chunk ->
            for (yy in chunk) {
                val start = c.starts[yy]
                val count = c.counts[yy]
                val kk = yy * c.kernelSize
                for (x in 0 until image.width) {
                    var r = half
                    var g = half
                    var b = half
                    for (y in 0 until count) {
                        val p = (start + y) * stride + x * 3
                        val w = k[kk + y]
                        r += (src[p].toInt() and 0xFF) * w
                        g += (src[p + 1].toInt() and 0xFF) * w
                        b += (src[p + 2].toInt() and 0xFF) * w
                    }
                    val o = (yy * image.width + x) * 3
                    out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b)
                }
            }
        }
        return RGBImage(image.width, height, out)
    }
}
