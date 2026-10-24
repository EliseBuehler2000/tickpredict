package com.ericbuehler.tickpredict.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.SelectableDates
import androidx.compose.material3.SheetValue
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.ericbuehler.tickpredict.PhotoLoader
import com.ericbuehler.tickpredict.app
import com.ericbuehler.tickpredict.inference.TickPrediction
import com.ericbuehler.tickpredict.model.BodyArea
import com.ericbuehler.tickpredict.model.Scan
import com.ericbuehler.tickpredict.model.confidenceSummary
import com.ericbuehler.tickpredict.model.title
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.UUID

/** Saves a new scan, or edits a saved one's notes. */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun SaveScanSheet(
    prediction: TickPrediction,
    imageData: ByteArray?,
    takenAt: Long,
    editing: Scan?,
    onDismiss: () -> Unit,
    onSave: (Scan) -> Unit,
) {
    val context = LocalContext.current
    val store = context.app.store
    val scope = rememberCoroutineScope()
    val haptics = rememberHaptics()

    // Field values; the originals tell whether anything changed before discarding.
    val original = remember {
        Fields(editing?.foundOn ?: takenAt, editing?.bodyArea, editing?.place ?: "", editing?.notes ?: "")
    }
    var foundOn by remember { mutableStateOf(original.foundOn) }
    var bodyArea by remember { mutableStateOf(original.bodyArea) }
    var place by remember { mutableStateOf(original.place) }
    var notes by remember { mutableStateOf(original.notes) }
    var saving by remember { mutableStateOf(false) }
    var saveFailed by remember { mutableStateOf(false) }
    var confirmDiscard by remember { mutableStateOf(false) }
    var pickDate by remember { mutableStateOf(false) }
    val hasChanges = Fields(foundOn, bodyArea, place, notes) != original
    val notesFocus = remember { FocusRequester() }
    var closing by remember { mutableStateOf(false) }

    val sheet = rememberModalBottomSheetState(
        skipPartiallyExpanded = true,
        confirmValueChange = { value ->
            // Swiping down with unsaved changes asks first.
            if (value == SheetValue.Hidden && !closing && Fields(foundOn, bodyArea, place, notes) != original) {
                confirmDiscard = true
                false
            } else {
                true
            }
        },
    )

    fun close() {
        closing = true
        scope.launch { sheet.hide() }.invokeOnCompletion { onDismiss() }
    }

    fun cancel() {
        if (hasChanges) confirmDiscard = true else close()
    }

    fun save() {
        if (saving) return
        saving = true
        scope.launch {
            val scan = (editing ?: Scan(UUID.randomUUID().toString(), takenAt, prediction, foundOn)).copy(
                foundOn = foundOn, bodyArea = bodyArea, place = place.trim(), notes = notes.trim(),
            )
            try {
                if (editing == null) store.insert(scan, imageData ?: error("No photo")) else store.update(scan)
            } catch (_: Exception) {
                saving = false
                saveFailed = true
                return@launch
            }
            onSave(scan)
            close()
        }
    }

    val thumbnail by produceState<ImageBitmap?>(null) {
        value = withContext(Dispatchers.IO) {
            val bytes = imageData?.let { PhotoLoader.thumbnailJpeg(it, 160) }
                ?: editing?.let { store.thumbnailFile(it.id).takeIf { file -> file.exists() }?.readBytes() }
            bytes?.let { android.graphics.BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
        }
    }

    ModalBottomSheet(
        onDismissRequest = { if (hasChanges) confirmDiscard = true else onDismiss() },
        sheetState = sheet,
        containerColor = Theme.ground,
        dragHandle = null,
    ) {
        Column(Modifier.fillMaxSize().statusBarsPadding().imePadding()) {
            Box(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp)) {
                TextButton(onClick = ::cancel, modifier = Modifier.align(Alignment.CenterStart)) {
                    Text("Cancel", style = Type.body, color = Theme.accent)
                }
                Text(if (editing == null) "Save scan" else "Edit scan", Modifier.align(Alignment.Center), style = Type.headline, color = Theme.ink)
                TextButton(onClick = ::save, enabled = !saving, modifier = Modifier.align(Alignment.CenterEnd)) {
                    Text(if (editing == null) "Save" else "Done", style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.accent)
                }
            }
            Column(
                Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
            ) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(Theme.surface)
                        .padding(horizontal = 14.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    ThumbnailFrame(thumbnail, 44.dp)
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(prediction.title, style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink)
                        Text(prediction.confidenceSummary, style = Type.footnote, color = Theme.muted)
                    }
                }

                Section("When") {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .heightIn(min = 50.dp)
                            .clip(RoundedCornerShape(16.dp))
                            .background(Theme.surface)
                            .clickable(role = Role.Button) { pickDate = true }
                            .padding(horizontal = 14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("Found", Modifier.weight(1f), style = Type.body, color = Theme.ink)
                        Text(
                            formatDate(foundOn),
                            Modifier
                                .clip(RoundedCornerShape(8.dp))
                                .background(Theme.track)
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                            style = Type.body,
                            color = Theme.ink,
                        )
                    }
                }

                Section("Where on the body") {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(16.dp))
                            .background(Theme.surface)
                            .padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(14.dp),
                    ) {
                        BodyOutline(bodyArea?.let { Offset(it.markerX, it.markerY) })
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            for (area in BodyArea.entries) {
                                Chip(area.title, bodyArea == area) {
                                    bodyArea = if (bodyArea == area) null else area
                                    haptics.selection()
                                }
                            }
                        }
                    }
                }

                Section("Where were you") {
                    Field(
                        place, { place = it }, "e.g. Backyard, hiking trail",
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences, imeAction = ImeAction.Next),
                        keyboardActions = KeyboardActions(onNext = { notesFocus.requestFocus() }),
                    )
                }

                Section("Notes") {
                    Field(
                        notes, { notes = it }, "Anything to tell your doctor",
                        modifier = Modifier.focusRequester(notesFocus),
                        minLines = 3,
                        maxLines = 8,
                        keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences),
                    )
                }

                Text(
                    "Scans are stored only on this phone.",
                    Modifier.fillMaxWidth(),
                    style = Type.caption,
                    color = Theme.muted,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }

    if (pickDate) {
        FoundDatePicker(foundOn, onPick = { foundOn = it; pickDate = false }, onDismiss = { pickDate = false })
    }

    if (confirmDiscard) {
        AlertDialog(
            onDismissRequest = { confirmDiscard = false },
            title = { Text("Discard your changes?") },
            confirmButton = {
                TextButton(onClick = { confirmDiscard = false; close() }) { Text("Discard changes", color = Theme.bite) }
            },
            dismissButton = { TextButton(onClick = { confirmDiscard = false }) { Text("Keep editing") } },
        )
    }

    if (saveFailed) {
        AlertDialog(
            onDismissRequest = { saveFailed = false },
            title = { Text("Couldn't save the scan") },
            text = { Text("Your phone may be low on storage. Free up some space and try again.") },
            confirmButton = { TextButton(onClick = { saveFailed = false }) { Text("OK") } },
        )
    }
}

private data class Fields(val foundOn: Long, val bodyArea: BodyArea?, val place: String, val notes: String)

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Eyebrow(title, Modifier.padding(start = 4.dp))
        content()
    }
}

@Composable
private fun Field(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    modifier: Modifier = Modifier,
    singleLine: Boolean = false,
    minLines: Int = 1,
    maxLines: Int = Int.MAX_VALUE,
    keyboardOptions: KeyboardOptions = KeyboardOptions.Default,
    keyboardActions: KeyboardActions = KeyboardActions.Default,
) {
    TextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier.fillMaxWidth(),
        placeholder = { Text(placeholder, style = Type.body, color = Theme.muted) },
        textStyle = Type.body.copy(color = Theme.ink),
        singleLine = singleLine,
        minLines = minLines,
        maxLines = if (singleLine) 1 else maxLines,
        keyboardOptions = keyboardOptions,
        keyboardActions = keyboardActions,
        shape = RoundedCornerShape(16.dp),
        colors = TextFieldDefaults.colors(
            focusedContainerColor = Theme.surface,
            unfocusedContainerColor = Theme.surface,
            focusedIndicatorColor = Color.Transparent,
            unfocusedIndicatorColor = Color.Transparent,
            cursorColor = Theme.accent,
        ),
    )
}

/** Picks the day the tick was found, keeping its time of day. Days after today are off unless already chosen. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FoundDatePicker(foundOn: Long, onPick: (Long) -> Unit, onDismiss: () -> Unit) {
    val zone = ZoneId.systemDefault()
    val current = Instant.ofEpochMilli(foundOn).atZone(zone)
    val latest = maxOf(current.toLocalDate(), LocalDate.now())
    val state = rememberDatePickerState(
        initialSelectedDateMillis = current.toLocalDate().atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli(),
        selectableDates = object : SelectableDates {
            override fun isSelectableDate(utcTimeMillis: Long) =
                Instant.ofEpochMilli(utcTimeMillis).atZone(ZoneOffset.UTC).toLocalDate() <= latest

            override fun isSelectableYear(year: Int) = year <= latest.year
        },
    )
    DatePickerDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = {
                val day = state.selectedDateMillis?.let { Instant.ofEpochMilli(it).atZone(ZoneOffset.UTC).toLocalDate() }
                if (day == null) onDismiss()
                else onPick(day.atTime(current.toLocalTime()).atZone(zone).toInstant().toEpochMilli())
            }) { Text("Done") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    ) {
        DatePicker(state, showModeToggle = false)
    }
}
