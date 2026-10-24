package com.ericbuehler.tickpredict.model

import com.ericbuehler.tickpredict.PhotoLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.IOException

/**
 * Saved scans, kept only on this device: an index of results and notes, plus
 * each photo and its thumbnail as files.
 */
class ScanStore(private val directory: File) {
    private val index = File(directory, "scans.json")
    private val mutex = Mutex()
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val _scans = MutableStateFlow(read())

    /** Every saved scan, in no particular order. */
    val scans: StateFlow<List<Scan>> = _scans.asStateFlow()

    fun scan(id: String): Scan? = _scans.value.firstOrNull { it.id == id }

    fun imageFile(id: String) = File(directory, "$id.photo")
    fun thumbnailFile(id: String) = File(directory, "$id.thumb.jpg")

    private fun read(): List<Scan> = try {
        if (index.exists()) json.decodeFromString<List<Scan>>(index.readText()) else emptyList()
    } catch (_: Exception) {
        emptyList()
    }

    /** Writes to a temporary file first, so a failed save never damages the index. */
    private fun write(scans: List<Scan>) {
        directory.mkdirs()
        val temporary = File(directory, "scans.json.tmp")
        temporary.writeText(json.encodeToString(scans))
        if (!temporary.renameTo(index)) throw IOException("Couldn't replace the scan index")
        _scans.value = scans
    }

    /** Saves a new scan with its photo. Throws if the device is out of space. */
    suspend fun insert(scan: Scan, imageData: ByteArray) = withContext(Dispatchers.IO) {
        mutex.withLock {
            try {
                directory.mkdirs()
                imageFile(scan.id).writeBytes(imageData)
                PhotoLoader.thumbnailJpeg(imageData, 240)?.let { thumbnailFile(scan.id).writeBytes(it) }
                write(_scans.value + scan)
            } catch (error: Exception) {
                imageFile(scan.id).delete()
                thumbnailFile(scan.id).delete()
                throw error
            }
        }
    }

    /** Replaces a saved scan's details. */
    suspend fun update(scan: Scan) = withContext(Dispatchers.IO) {
        mutex.withLock { write(_scans.value.map { if (it.id == scan.id) scan else it }) }
    }

    suspend fun delete(id: String) = withContext(Dispatchers.IO) {
        mutex.withLock {
            write(_scans.value.filter { it.id != id })
            imageFile(id).delete()
            thumbnailFile(id).delete()
        }
    }

    suspend fun deleteAll() = withContext(Dispatchers.IO) {
        mutex.withLock {
            val ids = _scans.value.map { it.id }
            write(emptyList())
            ids.forEach { imageFile(it).delete(); thumbnailFile(it).delete() }
        }
    }
}
