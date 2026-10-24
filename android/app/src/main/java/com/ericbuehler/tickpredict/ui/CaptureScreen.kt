package com.ericbuehler.tickpredict.ui

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.NoPhotography
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

private enum class CameraStatus { STARTING, RUNNING, DENIED, UNAVAILABLE }

// Frosted controls over the camera, standing in for Liquid Glass.
private val Glass = Color.White.copy(alpha = 0.18f)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun CaptureScreen(onClose: () -> Unit, onCapture: (ByteArray) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val haptics = rememberHaptics()
    var status by remember { mutableStateOf(CameraStatus.STARTING) }
    var flashOn by remember { mutableStateOf(false) }
    var capturing by remember { mutableStateOf(false) }
    var failure by remember { mutableStateOf<String?>(null) }
    val flash = remember { Animatable(0f) }
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    val capture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    // About 12 MP: plenty for a 384-pixel model input, and quick to decode.
                    .setResolutionStrategy(ResolutionStrategy(Size(4032, 3024), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                    .build(),
            )
            .build()
    }
    val picker = rememberPhotoPicker { onCapture(it.data) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        status = if (granted) CameraStatus.STARTING else CameraStatus.DENIED
        if (granted) scope.launch { status = startCamera(context, lifecycleOwner, previewView, capture) }
    }

    LaunchedEffect(Unit) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            status = startCamera(context, lifecycleOwner, previewView, capture)
        } else {
            permission.launch(Manifest.permission.CAMERA)
        }
    }
    DisposableEffect(Unit) {
        onDispose {
            val provider = ProcessCameraProvider.getInstance(context)
            provider.addListener({ provider.get().unbindAll() }, ContextCompat.getMainExecutor(context))
        }
    }
    LightStatusBarIcons()

    fun shoot() {
        if (status != CameraStatus.RUNNING || capturing) return
        capturing = true
        haptics.impact()
        scope.launch {
            flash.animateTo(1f, tween(80))
            capture.flashMode = if (flashOn) ImageCapture.FLASH_MODE_ON else ImageCapture.FLASH_MODE_OFF
            val data = try {
                takePhoto(context, capture)
            } catch (_: Exception) {
                null
            }
            flash.animateTo(0f, tween(200))
            capturing = false
            if (data != null) onCapture(data) else failure = "Couldn't take the photo. Try again."
        }
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black),
    ) {
        when (status) {
            CameraStatus.STARTING, CameraStatus.RUNNING -> Viewfinder(previewView)
            CameraStatus.DENIED -> Unavailable(
                "Camera access is off",
                "Allow camera access in Settings, or choose a photo you already took.",
                settings = true,
            )
            CameraStatus.UNAVAILABLE -> Unavailable(
                "Camera isn't available",
                "Choose a photo of the tick from your library instead.",
                settings = false,
            )
        }

        Column(
            Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(horizontal = 20.dp),
        ) {
            Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                RoundGlassButton("Close camera", onClose) {
                    Icon(Icons.Filled.Close, contentDescription = null, tint = Color.White)
                }
                Spacer(Modifier.weight(1f))
                if (status == CameraStatus.RUNNING) {
                    RoundGlassButton("Flash", { flashOn = !flashOn }, stateDescription = if (flashOn) "On" else "Off") {
                        Icon(
                            if (flashOn) Icons.Filled.FlashOn else Icons.Filled.FlashOff,
                            contentDescription = null,
                            tint = if (flashOn) Color(0xFFFFD60A) else Color.White,
                        )
                    }
                }
            }
            Spacer(Modifier.weight(1f))
            Column(Modifier.padding(bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
                FlowRow(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    for (tip in listOf("White paper", "Bright light", "Get close")) {
                        Text(
                            tip,
                            Modifier
                                .clip(CircleShape)
                                .background(Glass)
                                .padding(horizontal = 12.dp, vertical = 7.dp),
                            style = Type.footnote.copy(fontWeight = FontWeight.Medium),
                            color = Color.White,
                        )
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) {
                        Box(
                            Modifier
                                .size(52.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .background(Glass)
                                .clickable(role = Role.Button) { picker.choose() }
                                .semantics { contentDescription = "Choose from Photos" },
                            contentAlignment = Alignment.Center,
                        ) {
                            Icon(Icons.Filled.PhotoLibrary, contentDescription = null, tint = Color.White)
                        }
                    }
                    val shutterScale by animateFloatAsState(if (capturing) 0.9f else 1f, label = "shutter")
                    Box(
                        Modifier
                            .size(76.dp)
                            .alpha(if (status == CameraStatus.RUNNING) 1f else 0.4f)
                            .border(4.dp, Color.White, CircleShape)
                            .padding(7.dp)
                            .scale(shutterScale)
                            .clip(CircleShape)
                            .background(Color.White)
                            .clickable(enabled = status == CameraStatus.RUNNING, role = Role.Button) { shoot() }
                            .semantics { contentDescription = "Take photo" },
                    )
                    Text(
                        "8 common U.S. ticks",
                        Modifier.weight(1f).alpha(0.75f),
                        style = Type.caption,
                        color = Color.White,
                        textAlign = TextAlign.End,
                    )
                }
            }
        }

        if (flash.value > 0f) Box(Modifier.fillMaxSize().alpha(flash.value).background(Color.White))
        PhotoPickerOverlay(picker)
    }

    failure?.let { message ->
        AlertDialog(
            onDismissRequest = { failure = null },
            confirmButton = { TextButton(onClick = { failure = null }) { Text("OK") } },
            title = { Text(message) },
        )
    }
}

@Composable
private fun Viewfinder(previewView: PreviewView) {
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val reticle = 250.dp
        val centerY = maxHeight * 0.42f
        AndroidView({ previewView }, Modifier.fillMaxSize())
        // Dim everything outside the framing circle.
        Canvas(Modifier.fillMaxSize().graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)) {
            val center = Offset(size.width / 2, centerY.toPx())
            val radius = reticle.toPx() / 2
            drawRect(Color.Black.copy(alpha = 0.38f))
            drawCircle(Color.Black, radius, center, blendMode = BlendMode.Clear)
            drawCircle(Color.White.copy(alpha = 0.9f), radius - 1.dp.toPx(), center, style = Stroke(2.dp.toPx()))
        }
        Text(
            "Center the tick in the circle",
            Modifier
                .align(Alignment.TopCenter)
                .offset(y = centerY + reticle / 2 + 18.dp)
                .clip(CircleShape)
                .background(Glass)
                .padding(horizontal = 14.dp, vertical = 8.dp),
            style = Type.subheadline.copy(fontWeight = FontWeight.SemiBold),
            color = Color.White,
        )
    }
}

@Composable
private fun Unavailable(title: String, message: String, settings: Boolean) {
    val context = LocalContext.current
    Column(
        Modifier
            .fillMaxSize()
            .padding(40.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(Icons.Filled.NoPhotography, contentDescription = null, tint = Color.White, modifier = Modifier.size(40.dp))
        Text(title, style = Type.rounded(Type.title3), color = Color.White)
        Text(message, style = Type.subheadline, color = Color.White.copy(alpha = 0.8f), textAlign = TextAlign.Center)
        if (settings) {
            Text(
                "Open Settings",
                Modifier
                    .padding(top = 4.dp)
                    .clip(CircleShape)
                    .background(Glass)
                    .clickable(role = Role.Button) { openAppSettings(context) }
                    .padding(horizontal = 18.dp, vertical = 10.dp),
                style = Type.headline,
                color = Color.White,
            )
        }
    }
}

@Composable
private fun RoundGlassButton(label: String, onClick: () -> Unit, stateDescription: String? = null, content: @Composable () -> Unit) {
    Box(
        Modifier
            .size(44.dp)
            .clip(CircleShape)
            .background(Glass)
            .clickable(role = Role.Button, onClick = onClick)
            .semantics {
                contentDescription = label
                stateDescription?.let { this.stateDescription = it }
            },
        contentAlignment = Alignment.Center,
    ) { content() }
}

/** White status bar icons over the black camera, restored when the camera closes. */
@Composable
private fun LightStatusBarIcons() {
    val view = LocalView.current
    DisposableEffect(view) {
        val window = (view.context as? Activity)?.window
        val controller = window?.let { WindowCompat.getInsetsController(it, view) }
        val wasLight = controller?.isAppearanceLightStatusBars
        controller?.isAppearanceLightStatusBars = false
        onDispose { if (wasLight != null) controller.isAppearanceLightStatusBars = wasLight }
    }
}

fun openAppSettings(context: Context) {
    context.startActivity(
        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
    )
}

private suspend fun startCamera(
    context: Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    previewView: PreviewView,
    capture: ImageCapture,
): CameraStatus = try {
    val provider = suspendCancellableCoroutine { continuation ->
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                continuation.resume(future.get())
            } catch (error: Exception) {
                continuation.resumeWithException(error)
            }
        }, ContextCompat.getMainExecutor(context))
    }
    if (!provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) {
        CameraStatus.UNAVAILABLE
    } else {
        val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
        CameraStatus.RUNNING
    }
} catch (_: Exception) {
    CameraStatus.UNAVAILABLE
}

/** Saves through a file so CameraX writes the EXIF orientation, then returns the JPEG. */
private suspend fun takePhoto(context: Context, capture: ImageCapture): ByteArray {
    val file = File(context.cacheDir, "capture.jpg")
    suspendCancellableCoroutine { continuation ->
        capture.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) = continuation.resume(Unit)
                override fun onError(exception: ImageCaptureException) = continuation.resumeWithException(exception)
            },
        )
    }
    return withContext(Dispatchers.IO) { file.readBytes().also { file.delete() } }
}
