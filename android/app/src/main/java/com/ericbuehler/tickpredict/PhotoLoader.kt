package com.ericbuehler.tickpredict

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import com.ericbuehler.tickpredict.inference.ImageDecoding
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlin.math.max

/**
 * Downsampled, orientation-corrected images for display. Model input uses
 * the full-resolution original instead (see ImageDecoding).
 */
object PhotoLoader {
    fun image(data: ByteArray, maxPixelSize: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(data, 0, data.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxPixelSize) sample *= 2
        val decoded = BitmapFactory.decodeByteArray(data, 0, data.size,
            BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
        val scale = minOf(1f, maxPixelSize.toFloat() / max(decoded.width, decoded.height))
        val matrix = orientationMatrix(ImageDecoding.orientation(data)).apply { preScale(scale, scale) }
        if (matrix.isIdentity) return decoded
        val oriented = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        if (oriented !== decoded) decoded.recycle()
        return oriented
    }

    fun thumbnailJpeg(data: ByteArray, maxPixelSize: Int): ByteArray? {
        val bitmap = image(data, maxPixelSize) ?: return null
        return ByteArrayOutputStream().use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 80, output)
            output.toByteArray()
        }
    }

    /**
     * When the photo was taken, from its EXIF data, so library photos keep their own date.
     * Without a recorded offset the time is local, as on iOS (ExifInterface would assume UTC).
     */
    fun captureDate(data: ByteArray): Long? = try {
        val exif = ExifInterface(ByteArrayInputStream(data))
        val original = exif.getAttribute(ExifInterface.TAG_DATETIME_ORIGINAL)
        val local = original?.let { LocalDateTime.parse(it.trim(), DateTimeFormatter.ofPattern("yyyy:MM:dd HH:mm:ss")) }
        val offset = exif.getAttribute(ExifInterface.TAG_OFFSET_TIME_ORIGINAL)?.let { ZoneOffset.of(it.trim()) }
        local?.atZone(offset ?: ZoneId.systemDefault())?.toInstant()?.toEpochMilli()
    } catch (_: Exception) {
        null
    }

    private fun orientationMatrix(orientation: Int) = Matrix().apply {
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> postScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { postRotate(90f); postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> { postRotate(-90f); postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(-90f)
        }
    }
}
