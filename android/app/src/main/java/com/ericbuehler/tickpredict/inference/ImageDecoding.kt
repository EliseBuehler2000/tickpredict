package com.ericbuehler.tickpredict.inference

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer

/** Decodes photos for the models, matching Pillow's `exif_transpose` and RGB conversion. */
object ImageDecoding {
    /** Above this, photos are decoded at half size so the decode fits in memory. */
    private const val MAX_PIXELS = 30_000_000

    /** Decode image data (JPEG, HEIC, PNG…) and apply its EXIF orientation. */
    fun decode(data: ByteArray): RGBImage {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(data, 0, data.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw ImagePreparationException("Unreadable image")
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inSampleSize = if (bounds.outWidth.toLong() * bounds.outHeight > MAX_PIXELS) 2 else 1
        }
        val bitmap = BitmapFactory.decodeByteArray(data, 0, data.size, options)
            ?: throw ImagePreparationException("Unreadable image")
        return try {
            rawRGB(bitmap, orientation(data))
        } finally {
            bitmap.recycle()
        }
    }

    fun orientation(data: ByteArray): Int = try {
        ExifInterface(ByteArrayInputStream(data))
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    } catch (_: Exception) {
        ExifInterface.ORIENTATION_NORMAL
    }

    /**
     * Copies the bitmap's own bytes, so no colour conversion happens (as in Pillow),
     * then packs to RGB and applies the EXIF orientation in the same pass.
     */
    private fun rawRGB(bitmap: Bitmap, orientation: Int): RGBImage {
        val width = bitmap.width
        val height = bitmap.height
        val rgba = ByteBuffer.allocate(width * height * 4)
        bitmap.copyPixelsToBuffer(rgba)
        val src = rgba.array()
        val swaps = orientation in 5..8
        val outW = if (swaps) height else width
        val outH = if (swaps) width else height
        val rgb = ByteArray(width * height * 3)
        parallelChunks(outH) { rows ->
            for (y in rows) {
                for (x in 0 until outW) {
                    val from = sourceIndex(x, y, width, height, orientation) * 4
                    val to = (y * outW + x) * 3
                    rgb[to] = src[from]; rgb[to + 1] = src[from + 1]; rgb[to + 2] = src[from + 2]
                }
            }
        }
        return RGBImage(outW, outH, rgb)
    }

    /**
     * Index of the source pixel for output pixel (x, y), matching `ImageOps.exif_transpose`
     * for orientations 1–8. `w` and `h` are the source's size.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun sourceIndex(x: Int, y: Int, w: Int, h: Int, orientation: Int): Int = when (orientation) {
        2 -> y * w + (w - 1 - x)                // mirror horizontal
        3 -> (h - 1 - y) * w + (w - 1 - x)      // rotate 180
        4 -> (h - 1 - y) * w + x                // mirror vertical
        5 -> x * w + y                          // transpose
        6 -> (h - 1 - x) * w + y                // rotate 90 clockwise
        7 -> (h - 1 - x) * w + (w - 1 - y)      // transverse
        8 -> x * w + (w - 1 - y)                // rotate 90 counter-clockwise
        else -> y * w + x
    }
}
