package com.ericbuehler.tickpredict.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.BitmapDrawable
import android.location.Location
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.CancellationSignal
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Directions
import androidx.compose.material.icons.filled.MyLocation
import androidx.compose.material.icons.filled.Phone
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.CustomZoomButtonsController
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.CopyrightOverlay
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.TilesOverlay
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Locale
import kotlin.coroutines.resume

/**
 * Finds nearby care with OpenStreetMap data: Photon for places, Nominatim for phone
 * numbers and typed towns or ZIP codes. The location is used for the search only and never stored.
 */
class CareFinder {
    enum class Kind(val title: String) { URGENT("Urgent care"), DOCTORS("Doctors"), HOSPITALS("Hospitals") }

    enum class Status {
        LOCATING, SEARCHING, READY,
        /** Location permission is off. */
        DENIED,
        /** No location fix arrived in time, or location is turned off. */
        NO_LOCATION,
        /** A typed town or ZIP code wasn't recognised. */
        PLACE_NOT_FOUND,
        /** OpenStreetMap couldn't be reached. */
        OFFLINE,
    }

    data class Place(
        val name: String,
        val latitude: Double,
        val longitude: Double,
        val distance: Double,
        val address: String?,
        val phone: String?,
        /** OpenStreetMap id such as "W443493239", for looking up the phone number. */
        val osmId: String?,
    )

    var kind by mutableStateOf(Kind.URGENT)
    var status by mutableStateOf(Status.LOCATING)
        private set
    var origin by mutableStateOf<GeoPoint?>(null)
        private set
    var places by mutableStateOf<List<Place>>(emptyList())
        private set
    private var searchId = 0

    suspend fun locate(context: Context) {
        status = Status.LOCATING
        if (!hasLocationPermission(context)) {
            status = Status.DENIED
            return
        }
        val found = currentLocation(context)
        if (found == null) {
            status = Status.NO_LOCATION
            return
        }
        origin = GeoPoint(found.latitude, found.longitude)
        search()
    }

    /** Searches around a typed town or ZIP code instead of the current location. */
    suspend fun useTyped(text: String) {
        status = Status.SEARCHING
        try {
            val url = "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=us&q=" +
                URLEncoder.encode(text, "UTF-8")
            val first = Json.parseToJsonElement(get(url)).jsonArray.firstOrNull()?.jsonObject
            val latitude = first?.get("lat")?.jsonPrimitive?.contentOrNull?.toDoubleOrNull()
            val longitude = first?.get("lon")?.jsonPrimitive?.contentOrNull?.toDoubleOrNull()
            if (latitude == null || longitude == null) {
                status = Status.PLACE_NOT_FOUND
                return
            }
            origin = GeoPoint(latitude, longitude)
            search()
        } catch (_: IOException) {
            status = Status.OFFLINE
        } catch (_: Exception) {
            status = Status.PLACE_NOT_FOUND
        }
    }

    suspend fun search() {
        val origin = origin ?: return
        // Switching the care type quickly overlaps searches; only the latest may update.
        searchId += 1
        val id = searchId
        status = Status.SEARCHING
        try {
            val found = nearby(kind, origin)
                .filter { !isForAnimals(it) && (kind != Kind.URGENT || isUrgentCare(it)) }
                .mapNotNull { place(it, origin) }
                .distinctBy { it.name.lowercase() to (it.latitude * 1000).toInt() }
                .sortedBy { it.distance }
                .take(8)
            val withPhones = withPhoneNumbers(found)
            if (id != searchId) return
            places = withPhones
            status = Status.READY
        } catch (_: Exception) {
            if (id != searchId) return
            places = emptyList()
            status = Status.OFFLINE
        }
    }

    /** Places of the kind's OpenStreetMap categories around the origin, from Photon's reverse search. */
    private suspend fun nearby(kind: Kind, origin: GeoPoint): List<JsonObject> {
        val tags = when (kind) {
            Kind.URGENT -> listOf("amenity:clinic", "amenity:doctors", "healthcare:urgent_care", "healthcare:clinic")
            Kind.DOCTORS -> listOf("amenity:doctors", "amenity:clinic", "healthcare:doctor", "healthcare:clinic")
            Kind.HOSPITALS -> listOf("amenity:hospital", "healthcare:hospital")
        }
        val url = "https://photon.komoot.io/reverse?lat=${origin.latitude}&lon=${origin.longitude}" +
            "&radius=$RADIUS_KM&limit=50&lang=en" + tags.joinToString("") { "&osm_tag=$it" }
        return Json.parseToJsonElement(get(url)).jsonObject["features"]?.jsonArray?.map { it.jsonObject } ?: emptyList()
    }

    private fun place(feature: JsonObject, origin: GeoPoint): Place? {
        val properties = feature["properties"]?.jsonObject ?: return null
        fun property(key: String) = properties[key]?.jsonPrimitive?.contentOrNull?.trim()?.ifEmpty { null }
        val name = property("name") ?: return null
        val coordinates = feature["geometry"]?.jsonObject?.get("coordinates")?.jsonArray ?: return null
        val longitude = coordinates.getOrNull(0)?.jsonPrimitive?.doubleOrNull ?: return null
        val latitude = coordinates.getOrNull(1)?.jsonPrimitive?.doubleOrNull ?: return null
        val distance = FloatArray(1)
        Location.distanceBetween(origin.latitude, origin.longitude, latitude, longitude, distance)
        val street = listOfNotNull(property("housenumber"), property("street")).joinToString(" ").ifEmpty { null }
        val address = listOfNotNull(street, property("city")).joinToString(", ").ifEmpty { null }
        val osmId = property("osm_type")?.let { type -> property("osm_id")?.let { "$type$it" } }
        return Place(name, latitude, longitude, distance[0].toDouble(), address, phone = null, osmId = osmId)
    }

    /** Adds phone numbers from Nominatim in one lookup. Places keep no number if it fails. */
    private suspend fun withPhoneNumbers(places: List<Place>): List<Place> {
        val ids = places.mapNotNull { it.osmId }
        if (ids.isEmpty()) return places
        val phones = try {
            val url = "https://nominatim.openstreetmap.org/lookup?format=json&extratags=1&osm_ids=" + ids.joinToString(",")
            Json.parseToJsonElement(get(url)).jsonArray.associate { element ->
                val result = element.jsonObject
                val type = result["osm_type"]?.jsonPrimitive?.contentOrNull?.take(1)?.uppercase()
                val tags = result["extratags"] as? JsonObject
                val phone = (tags?.get("phone") ?: tags?.get("contact:phone"))?.jsonPrimitive?.contentOrNull
                "$type${result["osm_id"]?.jsonPrimitive?.contentOrNull}" to phone?.split(";")?.first()?.trim()
            }
        } catch (_: Exception) {
            emptyMap()
        }
        return places.map { place -> place.osmId?.let { phones[it] }?.let { place.copy(phone = it) } ?: place }
    }

    /** Urgent care has no category of its own in most of OpenStreetMap, so it's found by name. */
    private fun isUrgentCare(feature: JsonObject): Boolean {
        val properties = feature["properties"]?.jsonObject ?: return false
        if (properties["osm_value"]?.jsonPrimitive?.contentOrNull == "urgent_care") return true
        val name = properties["name"]?.jsonPrimitive?.contentOrNull ?: return false
        return URGENT_NAME.containsMatchIn(name)
    }

    /** "Urgent care" searches can turn up vets; TickPredict is for people. */
    private fun isForAnimals(feature: JsonObject): Boolean {
        val properties = feature["properties"]?.jsonObject ?: return false
        if (properties["osm_value"]?.jsonPrimitive?.contentOrNull == "veterinary") return true
        val name = properties["name"]?.jsonPrimitive?.contentOrNull?.lowercase() ?: return false
        val words = name.split(Regex("[^\\p{L}]+"))
        return listOf("pet", "vet", "animal").any { it in words } || "veterinar" in name
    }

    private suspend fun get(url: String): String = withContext(Dispatchers.IO) {
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 25_000
            connection.setRequestProperty("User-Agent", USER_AGENT)
            if (connection.responseCode !in 200..299) throw IOException("HTTP ${connection.responseCode}")
            connection.inputStream.bufferedReader().use { it.readText() }
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        private const val RADIUS_KM = 20
        private const val USER_AGENT = "TickPredict/2.0 (Android; tick bite care finder)"
        private val URGENT_NAME = Regex("urgent|walk.?in|immediate|express ?care|convenient ?care|minute ?clinic|emergency", RegexOption.IGNORE_CASE)

        fun hasLocationPermission(context: Context) =
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

        /** One location fix, waiting up to 20 seconds. */
        @SuppressLint("MissingPermission")
        private suspend fun currentLocation(context: Context): Location? {
            val manager = context.getSystemService(LocationManager::class.java) ?: return null
            val provider = buildList {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) add(LocationManager.FUSED_PROVIDER)
                add(LocationManager.NETWORK_PROVIDER)
                add(LocationManager.GPS_PROVIDER)
            }.firstOrNull { manager.isProviderEnabled(it) } ?: return null
            val fix = withTimeoutOrNull(20_000) {
                suspendCancellableCoroutine { continuation ->
                    val signal = CancellationSignal()
                    continuation.invokeOnCancellation { signal.cancel() }
                    LocationManagerCompat.getCurrentLocation(manager, provider, signal, ContextCompat.getMainExecutor(context)) {
                        continuation.resume(it)
                    }
                }
            }
            return fix ?: manager.getLastKnownLocation(provider)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FindCareScreen(finder: CareFinder, onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var typed by rememberSaveable { mutableStateOf("") }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        scope.launch { finder.locate(context) }
    }

    fun useMyLocation() {
        if (CareFinder.hasLocationPermission(context)) {
            scope.launch { finder.locate(context) }
        } else {
            permission.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        }
    }

    fun searchTyped() {
        val text = typed.trim()
        if (text.isNotEmpty()) scope.launch { finder.useTyped(text) }
    }

    LaunchedEffect(Unit) {
        // Runs again when returning to this screen; keep earlier results.
        if (finder.origin == null) useMyLocation()
    }

    Column(Modifier.fillMaxSize()) {
        InlineTopBar("Find care", onBack)
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp)
                .padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(240.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(Theme.photoBackdrop),
            ) {
                CareMap(finder)
                Box(
                    Modifier
                        .align(Alignment.TopEnd)
                        .padding(10.dp)
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(Theme.surface.copy(alpha = 0.92f))
                        .clickable(role = Role.Button, onClick = ::useMyLocation)
                        .semantics { contentDescription = "Use my location" },
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(Icons.Filled.MyLocation, contentDescription = null, tint = Theme.accent, modifier = Modifier.size(20.dp))
                }
            }

            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                CareFinder.Kind.entries.forEachIndexed { index, kind ->
                    SegmentedButton(
                        selected = finder.kind == kind,
                        onClick = {
                            if (finder.kind != kind) {
                                finder.kind = kind
                                scope.launch { finder.search() }
                            }
                        },
                        shape = SegmentedButtonDefaults.itemShape(index, CareFinder.Kind.entries.size),
                        icon = {},
                        colors = SegmentedButtonDefaults.colors(
                            activeContainerColor = Theme.surface,
                            activeContentColor = Theme.ink,
                            inactiveContainerColor = Theme.track,
                            inactiveContentColor = Theme.ink,
                            activeBorderColor = Theme.line,
                            inactiveBorderColor = Theme.line,
                        ),
                    ) {
                        Text(kind.title, style = Type.footnote.copy(fontWeight = if (finder.kind == kind) FontWeight.SemiBold else FontWeight.Normal), maxLines = 1)
                    }
                }
            }

            CareContent(finder, typed, { typed = it }, ::searchTyped, ::useMyLocation, retry = {
                scope.launch { if (finder.origin == null) finder.locate(context) else finder.search() }
            })

            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Theme.warningFill)
                    .clickable(role = Role.Button) { dial(context, "911") }
                    .semantics(mergeDescendants = true) {
                        contentDescription = "Emergency: trouble breathing, a stiff neck, or facial drooping. Call 911."
                    }
                    .padding(14.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Icon(Icons.Filled.Warning, contentDescription = null, tint = Theme.warningInk, modifier = Modifier.size(20.dp))
                Text(
                    buildAnnotatedString {
                        append("Trouble breathing, a stiff neck, or facial drooping? ")
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append("Call 911.") }
                    },
                    Modifier.weight(1f),
                    style = Type.subheadline,
                    color = Theme.warningInk,
                )
                Icon(Icons.Filled.Phone, contentDescription = null, tint = Theme.warningInk, modifier = Modifier.size(20.dp))
            }

            Text(
                "Places from OpenStreetMap. Your location is used only for this search and is never stored.",
                Modifier.fillMaxWidth(),
                style = Type.caption,
                color = Theme.muted,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun CareContent(
    finder: CareFinder,
    typed: String,
    onTyped: (String) -> Unit,
    searchTyped: () -> Unit,
    useMyLocation: () -> Unit,
    retry: () -> Unit,
) {
    val context = LocalContext.current
    when (finder.status) {
        CareFinder.Status.LOCATING, CareFinder.Status.SEARCHING -> Row(
            Modifier.fillMaxWidth().heightIn(min = 80.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterHorizontally),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CircularProgressIndicator(Modifier.size(20.dp), color = Theme.muted, strokeWidth = 2.dp)
            Text(
                if (finder.status == CareFinder.Status.LOCATING) "Finding your location" else "Searching nearby",
                style = Type.body, color = Theme.muted,
            )
        }
        CareFinder.Status.OFFLINE -> Card {
            Text("Couldn't reach OpenStreetMap", style = Type.headline, color = Theme.ink)
            Text("Check your internet connection and try again.", style = Type.subheadline, color = Theme.body)
            ProminentButton(retry) { Text("Try again", style = Type.headline) }
        }
        CareFinder.Status.DENIED, CareFinder.Status.NO_LOCATION, CareFinder.Status.PLACE_NOT_FOUND -> Card {
            Text(
                if (finder.status == CareFinder.Status.PLACE_NOT_FOUND) "Couldn't find that place" else "Search by town instead",
                style = Type.headline, color = Theme.ink,
            )
            Text(
                when (finder.status) {
                    CareFinder.Status.DENIED -> "Location access is off. Type a town or ZIP code to search there."
                    CareFinder.Status.PLACE_NOT_FOUND -> "Check the spelling, or try a ZIP code."
                    else -> "Your location isn't available right now. Type a town or ZIP code to search there."
                },
                style = Type.subheadline, color = Theme.body,
            )
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TextField(
                    value = typed,
                    onValueChange = onTyped,
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Town or ZIP code", style = Type.body, color = Theme.muted) },
                    singleLine = true,
                    textStyle = Type.body.copy(color = Theme.ink),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                    keyboardActions = KeyboardActions(onSearch = { searchTyped() }),
                    shape = RoundedCornerShape(12.dp),
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Theme.ground,
                        unfocusedContainerColor = Theme.ground,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        cursorColor = Theme.accent,
                    ),
                )
                ProminentButton(searchTyped) { Text("Search", style = Type.headline) }
            }
            if (finder.status == CareFinder.Status.DENIED) {
                LinkButton("Turn on location in Settings", { openAppSettings(context) },
                    style = Type.subheadline.copy(fontWeight = FontWeight.SemiBold))
            } else if (finder.status == CareFinder.Status.NO_LOCATION) {
                LinkButton("Try my location again", useMyLocation, style = Type.subheadline.copy(fontWeight = FontWeight.SemiBold))
            }
        }
        CareFinder.Status.READY -> if (finder.places.isEmpty()) {
            Card {
                Text("Nothing found nearby", style = Type.headline, color = Theme.ink)
                Text("Try another type of care, or call your doctor's office.", style = Type.subheadline, color = Theme.body)
            }
        } else {
            Column(
                Modifier
                    .clip(RoundedCornerShape(18.dp))
                    .background(Theme.surface),
            ) {
                finder.places.forEachIndexed { index, place ->
                    if (index > 0) Divider(Modifier.padding(start = 54.dp))
                    PlaceRow(index + 1, place)
                }
            }
        }
    }
}

@Composable
private fun PlaceRow(number: Int, place: CareFinder.Place) {
    val context = LocalContext.current
    Row(
        Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Box(
            Modifier
                .size(28.dp)
                .clip(CircleShape)
                .background(Theme.accentTint),
            contentAlignment = Alignment.Center,
        ) {
            Text("$number", style = Type.footnote.copy(fontWeight = FontWeight.Bold), color = Theme.accent)
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(place.name, style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(
                listOfNotNull(formatDistance(place.distance), place.address).joinToString(" · "),
                style = Type.footnote, color = Theme.muted, maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
        }
        place.phone?.let { phone ->
            Box(
                Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Theme.accentTint)
                    .clickable(role = Role.Button) { dial(context, phone) }
                    .semantics { contentDescription = "Call ${place.name}" },
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Filled.Phone, contentDescription = null, tint = Theme.accent, modifier = Modifier.size(18.dp))
            }
        }
        Box(
            Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(Theme.accent)
                .clickable(role = Role.Button) { directions(context, place) }
                .semantics { contentDescription = "Directions to ${place.name}" },
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Filled.Directions, contentDescription = null, tint = Theme.onAccent, modifier = Modifier.size(20.dp))
        }
    }
}

/** OpenStreetMap tiles with the search origin and numbered places, framed to fit them all. */
@Composable
private fun CareMap(finder: CareFinder) {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val dark = Theme.isDark
    val accent = Theme.accent.toArgb()
    val onAccent = Theme.onAccent.toArgb()
    val map = remember {
        MapView(context).apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
            minZoomLevel = 3.0
            controller.setZoom(4.0)
            controller.setCenter(GeoPoint(39.8, -98.6))
            overlays.add(CopyrightOverlay(context))
        }
    }
    DisposableEffect(lifecycle) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> map.onResume()
                Lifecycle.Event.ON_PAUSE -> map.onPause()
                else -> {}
            }
        }
        lifecycle.addObserver(observer)
        map.onResume()
        onDispose {
            lifecycle.removeObserver(observer)
            map.onPause()
        }
    }
    AndroidView({ map }, Modifier.fillMaxSize()) { view ->
        view.overlayManager.tilesOverlay.setColorFilter(if (dark) TilesOverlay.INVERT_COLORS else null)
        view.overlays.removeAll { it is Marker }
        val origin = finder.origin ?: return@AndroidView
        view.overlays.add(Marker(view).apply {
            position = origin
            icon = dot(context, android.graphics.Color.rgb(0, 122, 255))
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            setInfoWindow(null)
            title = "Search location"
        })
        finder.places.forEachIndexed { index, place ->
            view.overlays.add(Marker(view).apply {
                position = GeoPoint(place.latitude, place.longitude)
                icon = numbered(context, index + 1, accent, onAccent)
                setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                setInfoWindow(null)
                title = place.name
            })
        }
        val points = listOf(origin) + finder.places.map { GeoPoint(it.latitude, it.longitude) }
        view.post {
            if (points.size == 1) {
                view.controller.setZoom(13.0)
                view.controller.setCenter(origin)
            } else {
                view.zoomToBoundingBox(BoundingBox.fromGeoPointsSafe(points).increaseByScale(1.3f), false, 48)
            }
        }
        view.invalidate()
    }
}

private fun numbered(context: Context, number: Int, fill: Int, text: Int): BitmapDrawable {
    val density = context.resources.displayMetrics.density
    val size = (28 * density).toInt()
    val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = android.graphics.Canvas(bitmap)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    paint.color = android.graphics.Color.WHITE
    canvas.drawCircle(size / 2f, size / 2f, size / 2f, paint)
    paint.color = fill
    canvas.drawCircle(size / 2f, size / 2f, size / 2f - 2 * density, paint)
    paint.color = text
    paint.textSize = 13 * density
    paint.typeface = Typeface.DEFAULT_BOLD
    paint.textAlign = Paint.Align.CENTER
    canvas.drawText("$number", size / 2f, size / 2f - (paint.descent() + paint.ascent()) / 2, paint)
    return BitmapDrawable(context.resources, bitmap)
}

private fun dot(context: Context, fill: Int): BitmapDrawable {
    val density = context.resources.displayMetrics.density
    val size = (18 * density).toInt()
    val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = android.graphics.Canvas(bitmap)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    paint.color = android.graphics.Color.WHITE
    canvas.drawCircle(size / 2f, size / 2f, size / 2f, paint)
    paint.color = fill
    canvas.drawCircle(size / 2f, size / 2f, size / 2f - 3 * density, paint)
    return BitmapDrawable(context.resources, bitmap)
}

/** Road distance style: feet and miles in the U.S., metres and kilometres elsewhere. */
private fun formatDistance(meters: Double): String {
    val imperial = Locale.getDefault().country in setOf("US", "LR", "MM")
    return if (imperial) {
        val miles = meters / 1609.344
        when {
            miles < 0.1 -> "${(meters * 3.28084 / 10).toInt() * 10} ft"
            miles < 10 -> String.format(Locale.US, "%.1f mi", miles)
            else -> "${miles.toInt()} mi"
        }
    } else {
        if (meters < 1000) "${(meters / 10).toInt() * 10} m" else String.format(Locale.US, "%.1f km", meters / 1000)
    }
}

private fun dial(context: Context, number: String) {
    val digits = number.filter { it.isDigit() || it == '+' }
    if (digits.isEmpty()) return
    context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$digits")).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}

private fun directions(context: Context, place: CareFinder.Place) {
    val url = "https://www.google.com/maps/dir/?api=1&destination=${place.latitude},${place.longitude}"
    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}
