package com.ericbuehler.tickpredict.ui

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ericbuehler.tickpredict.PendingPhoto
import com.ericbuehler.tickpredict.app
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun HomeScreen(
    openCamera: () -> Unit,
    openHistory: () -> Unit,
    openGuide: () -> Unit,
    open: (PendingPhoto) -> Unit,
    openScan: (String) -> Unit,
) {
    val context = LocalContext.current
    val store = context.app.store
    val scans by store.scans.collectAsStateWithLifecycle()
    val recent = remember(scans) { scans.sortedByDescending { it.createdAt }.take(2) }
    val picker = rememberPhotoPicker(open)

    Box(Modifier.fillMaxSize()) {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            Column {
                LargeTitle("TickPredict")
                Text("Life stage and engorgement from one photo", style = Type.subheadline, color = Theme.muted)
            }

            ScanCard(openCamera = openCamera, choosePhoto = { picker.choose() })

            if (recent.isNotEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text("Recent", style = Type.rounded(Type.title3), color = Theme.ink, modifier = Modifier.weight(1f))
                        LinkButton("See all", openHistory)
                    }
                    Column(
                        Modifier
                            .clip(RoundedCornerShape(18.dp))
                            .background(Theme.surface),
                    ) {
                        recent.forEachIndexed { index, scan ->
                            if (index > 0) Divider(Modifier.padding(start = 74.dp))
                            ScanRow(scan, store.thumbnailFile(scan.id), Modifier.clickable { openScan(scan.id) })
                        }
                    }
                }
            }

            Card(fill = Theme.accentTint, padding = 16.dp) {
                Text("Just found a tick?", style = Type.rounded(Type.headline), color = Theme.ink)
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    NumberedLine(1, "Pull it straight out with fine tweezers")
                    NumberedLine(2, "Keep it in a sealed bag or taped to paper")
                    NumberedLine(3, "Scan it, then watch for symptoms for 30 days")
                }
                LinkButton("Read the full guide", openGuide, style = Type.subheadline.copy(fontWeight = FontWeight.SemiBold))
            }

            Text(
                "Knows 8 common U.S. ticks. Informational, not a diagnosis.",
                style = Type.caption,
                color = Theme.muted,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        PhotoPickerOverlay(picker)
    }
}

@Composable
private fun ScanCard(openCamera: () -> Unit, choosePhoto: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(Theme.scanCard)
            .padding(22.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            Box(
                Modifier
                    .size(52.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White.copy(alpha = 0.14f)),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Filled.CenterFocusWeak, contentDescription = null, tint = Color.White, modifier = Modifier.size(28.dp))
            }
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("Scan a tick", style = Type.rounded(Type.title2), color = Color.White)
                Text("Runs privately on your phone", style = Type.subheadline, color = Color.White.copy(alpha = 0.82f))
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                Modifier
                    .weight(1f)
                    .heightIn(min = 50.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.White)
                    .clickable(role = Role.Button, onClick = openCamera),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Filled.PhotoCamera, contentDescription = null, tint = Color(0xFF2E5A3A), modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Open camera", style = Type.headline, color = Color(0xFF2E5A3A))
            }
            Box(
                Modifier
                    .size(50.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.White.copy(alpha = 0.14f))
                    .clickable(role = Role.Button, onClick = choosePhoto)
                    .semantics { contentDescription = "Choose from Photos" },
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Filled.PhotoLibrary, contentDescription = null, tint = Color.White)
            }
        }
    }
}

/** The system photo picker, plus the "Opening photo" and failure states while a photo loads. */
class PhotoPicker {
    var loading by mutableStateOf(false)
    var failed by mutableStateOf(false)
    var choose: () -> Unit = {}
        internal set
}

@Composable
fun rememberPhotoPicker(onPhoto: (PendingPhoto) -> Unit): PhotoPicker {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val picker = remember { PhotoPicker() }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        picker.loading = true
        scope.launch {
            val data = readPhoto(context, uri)
            picker.loading = false
            if (data != null) onPhoto(PendingPhoto(data)) else picker.failed = true
        }
    }
    picker.choose = { launcher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    return picker
}

suspend fun readPhoto(context: Context, uri: Uri): ByteArray? = withContext(Dispatchers.IO) {
    try {
        context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
    } catch (_: Exception) {
        null
    }
}

@Composable
fun PhotoPickerOverlay(picker: PhotoPicker) {
    if (picker.loading) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(
                Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(Theme.surface.copy(alpha = 0.95f))
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                CircularProgressIndicator(color = Theme.accent)
                Text("Opening photo", style = Type.subheadline, color = Theme.ink)
            }
        }
    }
    if (picker.failed) {
        AlertDialog(
            onDismissRequest = { picker.failed = false },
            confirmButton = { TextButton(onClick = { picker.failed = false }) { Text("OK") } },
            title = { Text("Couldn't open that photo") },
            text = { Text("Try another photo, or check that it has finished downloading.") },
        )
    }
}
