package com.ericbuehler.tickpredict.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Inventory2
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.MoreHoriz
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.outlined.Circle
import androidx.compose.material.icons.outlined.VerifiedUser
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ericbuehler.tickpredict.PendingPhoto
import com.ericbuehler.tickpredict.PhotoLoader
import com.ericbuehler.tickpredict.ScanError
import com.ericbuehler.tickpredict.ScanException
import com.ericbuehler.tickpredict.app
import com.ericbuehler.tickpredict.inference.TaskPrediction
import com.ericbuehler.tickpredict.inference.TickPrediction
import com.ericbuehler.tickpredict.model.Engorgement
import com.ericbuehler.tickpredict.model.LifeStage
import com.ericbuehler.tickpredict.model.LymeRisk
import com.ericbuehler.tickpredict.model.Scan
import com.ericbuehler.tickpredict.model.Species
import com.ericbuehler.tickpredict.model.title
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.random.Random

sealed interface ResultSource {
    class New(val photo: PendingPhoto) : ResultSource
    class Saved(val scanId: String) : ResultSource
}

/** A result screen's state, kept with its route so it survives going to Find care and back. */
class ResultState {
    var prediction by mutableStateOf<TickPrediction?>(null)
    var failure by mutableStateOf<ScanError?>(null)
    var stepsDone by mutableIntStateOf(0)
    var savedScanId by mutableStateOf<String?>(null)
    var photo by mutableStateOf<ImageBitmap?>(null)
    var showDetails by mutableStateOf(false)
    var attempt by mutableIntStateOf(0)
    var deleting by mutableStateOf(false)
}

@Composable
fun ResultScreen(
    source: ResultSource,
    state: ResultState,
    scanAnother: () -> Unit,
    findCare: () -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val store = context.app.store
    val classifier = context.app.classifier
    val haptics = rememberHaptics()
    val view = LocalView.current
    val scope = rememberCoroutineScope()
    val scans by store.scans.collectAsStateWithLifecycle()
    val savedScan = state.savedScanId?.let { id -> scans.firstOrNull { it.id == id } }
    val isNew = source is ResultSource.New
    // A saved scan that was deleted, here or in another tab. A new scan whose saved copy
    // is deleted just becomes unsaved again.
    val isGone = state.deleting || (!isNew && savedScan == null)
    val prediction = if (isNew) state.prediction else savedScan?.prediction
    val takenAt = when (source) {
        is ResultSource.New -> source.photo.takenAt
        is ResultSource.Saved -> savedScan?.createdAt ?: 0L
    }
    var showSave by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }

    LaunchedEffect(savedScan == null) {
        if (isNew && state.savedScanId != null && savedScan == null) state.savedScanId = null
    }

    LaunchedEffect(state.attempt) {
        if (isGone) return@LaunchedEffect
        if (state.photo == null) {
            state.photo = withContext(Dispatchers.IO) {
                val data = when (source) {
                    is ResultSource.New -> source.photo.data
                    is ResultSource.Saved -> store.imageFile(source.scanId).takeIf { it.exists() }?.readBytes()
                }
                data?.let { PhotoLoader.image(it, 1400)?.asImageBitmap() }
            }
        }
        if (state.prediction != null || source !is ResultSource.New) return@LaunchedEffect
        state.failure = null
        state.stepsDone = 0
        // The models start straight away; the steps are revealed over about five
        // seconds to show what is being checked. A slow first run keeps the last
        // step spinning, and a failure shows at once.
        val durations = AnalysisSteps.durations()
        val result = coroutineScope {
            val reveal = launch {
                durations.dropLast(1).forEachIndexed { index, seconds ->
                    delay((seconds * 1000).toLong())
                    state.stepsDone = index + 1
                }
            }
            try {
                classifier.classify(source.photo).also { reveal.join() }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                reveal.cancel()
                state.failure = (error as? ScanException)?.error ?: ScanError.UNREADABLE_PHOTO
                null
            }
        } ?: return@LaunchedEffect
        delay((durations.last() * 1000).toLong())
        state.stepsDone = AnalysisSteps.steps.size
        delay(250)
        state.prediction = result
        haptics.success()
        view.announceForAccessibility("Result: ${result.title}")
    }

    Column(Modifier.fillMaxSize()) {
        InlineTopBar(if (isNew && savedScan == null) "Result" else "Scan", onBack) {
            if (!isGone && prediction != null) {
                if (savedScan == null) {
                    TextButton(onClick = { showSave = true }) {
                        Text("Save", style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.accent)
                    }
                } else {
                    Box {
                        IconButton(onClick = { menu = true }) {
                            Icon(Icons.Filled.MoreHoriz, contentDescription = "Scan options", tint = Theme.accent)
                        }
                        DropdownMenu(menu, onDismissRequest = { menu = false }, containerColor = Theme.surface) {
                            DropdownMenuItem(
                                text = { Text("Edit details") },
                                leadingIcon = { Icon(Icons.Filled.Edit, contentDescription = null) },
                                onClick = { menu = false; showSave = true },
                            )
                            DropdownMenuItem(
                                text = { Text("Delete scan", color = Theme.bite) },
                                leadingIcon = { Icon(Icons.Filled.Delete, contentDescription = null, tint = Theme.bite) },
                                onClick = { menu = false; confirmDelete = true },
                            )
                        }
                    }
                }
            }
        }

        if (isGone) {
            ContentUnavailable(Icons.Filled.Delete, "Scan deleted", "This scan is no longer in your history.")
        } else {
            Column(
                Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 32.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                PhotoHeader(state.photo, takenAt)
                when {
                    prediction != null -> Results(
                        prediction, savedScan, isNew,
                        showDetails = state.showDetails,
                        toggleDetails = { state.showDetails = !state.showDetails },
                        save = { showSave = true },
                        scanAnother = scanAnother,
                        findCare = findCare,
                    )
                    state.failure != null -> FailureCard(state.failure!!, retry = { state.attempt += 1 }, scanAnother = scanAnother)
                    else -> AnalysisSteps(state.stepsDone)
                }
            }
        }
    }

    if (showSave && prediction != null && !isGone) {
        SaveScanSheet(
            prediction = prediction,
            imageData = (source as? ResultSource.New)?.photo?.data,
            takenAt = takenAt,
            editing = savedScan,
            onDismiss = { showSave = false },
            onSave = { scan ->
                state.savedScanId = scan.id
                haptics.success()
            },
        )
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete this scan?") },
            text = { Text("The photo and your notes will be removed from this phone.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    val id = savedScan?.id ?: return@TextButton
                    state.deleting = true
                    scope.launch { store.delete(id) }
                    onBack()
                }) { Text("Delete scan", color = Theme.bite) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun FailureCard(failure: ScanError, retry: () -> Unit, scanAnother: () -> Unit) {
    Card {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(Icons.Filled.Warning, contentDescription = null, tint = Theme.ink)
            Text(
                if (failure == ScanError.MODELS_UNAVAILABLE) "Couldn't start the models" else "Couldn't read this photo",
                style = Type.headline, color = Theme.ink,
            )
        }
        Text(
            if (failure == ScanError.MODELS_UNAVAILABLE) "TickPredict's models didn't load. Try again, or close and reopen the app."
            else "This file couldn't be opened as a photo. Try another photo of the tick.",
            style = Type.subheadline, color = Theme.body,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SecondaryButton(retry) { Text("Try again", style = Type.headline) }
            ProminentButton(scanAnother) { Text("New photo", style = Type.headline) }
        }
    }
}

@Composable
private fun PhotoHeader(photo: ImageBitmap?, takenAt: Long) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(220.dp)
            .clip(RoundedCornerShape(22.dp))
            .background(Theme.photoBackdrop)
            .clearAndSetSemantics { contentDescription = "Tick photo, ${formatDate(takenAt)}" },
    ) {
        if (photo != null) {
            Image(photo, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        }
        if (takenAt > 0) {
            Text(
                formatDateTime(takenAt),
                Modifier
                    .align(Alignment.BottomStart)
                    .padding(12.dp)
                    .clip(CircleShape)
                    .background(Theme.surface.copy(alpha = 0.85f))
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                style = Type.footnote.copy(fontWeight = FontWeight.Medium),
                color = Theme.ink,
            )
        }
    }
}

@Composable
private fun Results(
    prediction: TickPrediction,
    scan: Scan?,
    isNew: Boolean,
    showDetails: Boolean,
    toggleDetails: () -> Unit,
    save: () -> Unit,
    scanAnother: () -> Unit,
    findCare: () -> Unit,
) {
    val stage = LifeStage(prediction.lifeStage.label)
    val engorgement = Engorgement(prediction.engorgement.label)

    prediction.species?.let { SpeciesCard(it, engorgement) }

    val stageCard = @Composable { modifier: Modifier ->
        AnswerCard("Life stage", stage.name, prediction.lifeStage.confidence, modifier) {
            SegmentScale(
                labels = LifeStage.order.map { LifeStage(it).shortName },
                colors = LifeStage.order.map { Theme.accent },
                selected = LifeStage.order.indexOf(stage.label).coerceAtLeast(0),
            )
        }
    }
    val engorgementCard = @Composable { modifier: Modifier ->
        AnswerCard("Engorgement", engorgement.name, prediction.engorgement.confidence, modifier) {
            SegmentScale(
                labels = Engorgement.order.map { Engorgement(it).shortName },
                colors = Theme.engorgementRamp,
                selected = engorgement.index,
                ramp = true,
            )
        }
    }
    if (LocalDensity.current.fontScale >= 1.5f) {
        stageCard(Modifier)
        engorgementCard(Modifier)
    } else {
        Row(Modifier.height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            stageCard(Modifier.weight(1f).fillMaxHeight())
            engorgementCard(Modifier.weight(1f).fillMaxHeight())
        }
    }

    LymeRisk.of(prediction)?.let { LymeCard(it, prediction, findCare) }

    val engorgementMeaning = if (stage.label == "adult male") {
        "Males stay small when they feed, so engorgement says little about how long it was attached."
    } else {
        engorgement.meaning
    }
    Card {
        Text("What this means", style = Type.rounded(Type.title3), color = Theme.ink)
        Text(boldLead(stage.name, stage.meaning), style = Type.subheadline, color = Theme.body)
        Text(boldLead(engorgement.name, engorgementMeaning), style = Type.subheadline, color = Theme.body)
    }

    scan?.let { ScanNotesCard(it) }

    NextSteps(findCare)

    Details(prediction, showDetails, toggleDetails)

    if (isNew) {
        Column(Modifier.padding(top = 4.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (scan == null) {
                ProminentButton(save, Modifier.fillMaxWidth()) { Text("Save to history", style = Type.headline) }
            } else {
                Row(
                    Modifier.fillMaxWidth().heightIn(min = 48.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = Theme.accent)
                    Spacer(Modifier.width(8.dp))
                    Text("Saved to History", style = Type.headline, color = Theme.accent)
                }
            }
            SecondaryButton(scanAnother, Modifier.fillMaxWidth()) { Text("Scan another", style = Type.headline) }
        }
    }

    Text(
        if (prediction.species == null) "Scanned with an earlier version that didn't identify species. Informational, not a diagnosis."
        else "Informational, not a diagnosis.",
        Modifier.fillMaxWidth().padding(top = 4.dp),
        style = Type.caption,
        color = Theme.muted,
        textAlign = TextAlign.Center,
    )
}

private fun boldLead(lead: String, rest: String) = buildAnnotatedString {
    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append("$lead.") }
    append(" $rest")
}

@Composable
private fun SpeciesCard(task: TaskPrediction, engorgement: Engorgement) {
    val species = Species(task.label)
    val runnerUp = task.ranked.drop(1).firstOrNull()
    val carries = species.carriesText
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(Theme.surface)
            .padding(18.dp)
            .semantics(mergeDescendants = true) {}
            .testTag("speciesCard"),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Eyebrow("Species")
        Row(verticalAlignment = Alignment.Top) {
            Text(species.name, Modifier.weight(1f), style = Type.rounded(Type.title2), color = Theme.ink)
            Spacer(Modifier.width(8.dp))
            Text(task.confidence.percent, Modifier.padding(top = 5.dp), style = Type.subheadline.copy(fontWeight = FontWeight.Bold), color = Theme.accent)
        }
        Text(
            buildAnnotatedString {
                species.otherName?.let { append("$it · ") }
                withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(species.label) }
            },
            style = Type.subheadline, color = Theme.muted,
        )
        ConfidenceBar(task.confidence)
        if (engorgement.label == "fully engorged") {
            Text(
                if (species.spreadsLyme) "Fully engorged ticks are harder to identify."
                else "Fully engorged ticks are harder to identify. It could still be a blacklegged tick.",
                style = Type.footnote, color = Theme.muted,
            )
        } else if (task.confidence < 0.6 && runnerUp != null) {
            Text(
                "Could also be a ${Species(runnerUp.first).name.lowercase(Locale.US)} (${runnerUp.second.percent}).",
                style = Type.footnote, color = Theme.muted,
            )
        }
        Divider()
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Can carry", style = Type.caption.copy(fontWeight = FontWeight.SemiBold), color = Theme.muted)
            Text(
                if (carries.isEmpty()) "No illness in people confirmed in the U.S."
                else carries.replaceFirstChar { it.uppercase() },
                style = Type.subheadline, color = Theme.ink,
            )
            species.note?.let { Text(it, style = Type.footnote, color = Theme.body) }
        }
    }
}

@Composable
private fun LymeCard(risk: LymeRisk, prediction: TickPrediction, findCare: () -> Unit) {
    val blacklegged = LymeRisk.blackleggedScore(prediction) ?: 0.0
    Card(fill = if (risk.isUrgent) Theme.warningFill else Theme.accentTint) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            val tint = if (risk.isUrgent) Theme.warningInk else Theme.ink
            Icon(if (risk.isUrgent) Icons.Filled.Error else Icons.Outlined.VerifiedUser, contentDescription = null, tint = tint)
            Text(risk.title, style = Type.rounded(Type.title3), color = tint)
        }
        Text(risk.explanation, style = Type.subheadline, color = Theme.body)
        if (risk != LymeRisk.NOT_LYME_TICK && blacklegged < 0.8) {
            Text(
                if (LymeRisk.isEngorged(prediction)) "Engorged ticks are hard to identify from a photo, so this treats it as a possible blacklegged tick."
                else "It may not be a blacklegged tick (${blacklegged.percent} likely), but this treats it as one to be safe.",
                style = Type.footnote, color = Theme.body,
            )
        }
        if (risk.isUrgent) FindCareButton(findCare)
    }
}

@Composable
fun FindCareButton(onClick: () -> Unit) {
    ProminentButton(onClick, Modifier.fillMaxWidth()) {
        Icon(Icons.Filled.LocationOn, contentDescription = null, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(8.dp))
        Text("Find care nearby", style = Type.headline)
    }
}

@Composable
private fun NextSteps(findCare: () -> Unit) {
    Card(fill = Theme.accentTint) {
        Text("What to do next", style = Type.rounded(Type.title3), color = Theme.ink)
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            StepRow(Icons.Filled.Inventory2, "Keep the tick", "Seal it in a bag or tape it to paper. Your doctor may want to see it.")
            StepRow(Icons.Filled.CalendarMonth, "Note the date and bite spot", "Save this scan and TickPredict keeps it for you.")
            StepRow(Icons.Filled.Visibility, "Watch for 30 days", "An expanding rash, fever, chills, or aches are reasons to call.")
            StepRow(Icons.Filled.MedicalServices, "Talk to a doctor", "Especially for an engorged tick. Show them this result.")
        }
        Box(Modifier.padding(top = 2.dp)) { FindCareButton(findCare) }
    }
}

@Composable
private fun Details(prediction: TickPrediction, expanded: Boolean, toggle: () -> Unit) {
    val rotation by animateFloatAsState(if (expanded) 180f else 0f, label = "chevron")
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(Theme.surface)
            .animateContentSize(),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .heightIn(min = 56.dp)
                .clickable(onClick = toggle)
                .semantics { stateDescription = if (expanded) "Expanded" else "Collapsed" }
                .padding(horizontal = 18.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Details", Modifier.weight(1f), style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink)
            Icon(Icons.Filled.ExpandMore, contentDescription = null, tint = Theme.muted, modifier = Modifier.rotate(rotation))
        }
        AnimatedVisibility(expanded, enter = fadeIn() + expandVertically(), exit = fadeOut() + shrinkVertically()) {
            Column {
                Divider()
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
                    prediction.species?.let { ScoreTable("Species scores", it) { label -> Species(label).name } }
                    ScoreTable("Life stage scores", prediction.lifeStage) { LifeStage(it).name }
                    ScoreTable("Engorgement scores", prediction.engorgement) { Engorgement(it).name }
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Eyebrow("Models · ${prediction.modelRelease}")
                        val style = Type.footnote
                        if (prediction.species == null) {
                            Text("Life stage: ConvNeXt Tiny, 320 px. 107 of 110 research photos correct.", style = style, color = Theme.body)
                        } else {
                            Text(
                                "Species and life stage: ConvNeXt Tiny, 384 px, averaged over 8 rotated views. On 1,304 held-out iNaturalist photos, species was right 85% of the time, and blacklegged vs. other ticks 97%. Life stage: 106 of 110 research photos correct.",
                                style = style, color = Theme.body,
                            )
                        }
                        Text("Engorgement: EfficientNet V2 S, 384 px, averaged over 8 rotated views (v1.0). 95 of 110 research photos correct.", style = style, color = Theme.body)
                        Text("Scores are uncalibrated model confidence, not the chance the answer is right. Runs entirely on this phone.", style = style, color = Theme.body)
                    }
                }
            }
        }
    }
}

/** What the models check, revealed one step at a time while a photo is analyzed. */
object AnalysisSteps {
    val steps = listOf(
        "Preparing the photo" to "Straightening, trimming, and resizing the photo",
        "Identifying the species" to "Using a trained ConvNeXt Tiny model",
        "Checking the life stage" to "Using a trained ConvNeXt Tiny model",
        "Measuring engorgement" to "Using a trained EfficientNet V2 S model",
        "Weighing Lyme risk" to "Combining all three results",
    )

    /** About five seconds in all, split unevenly so it doesn't feel mechanical. */
    fun durations(): List<Double> {
        val weights = steps.map { Random.nextDouble(0.6, 1.4) }
        val total = Random.nextDouble(4.5, 5.5)
        return weights.map { it / weights.sum() * total }
    }
}

@Composable
private fun AnalysisSteps(done: Int) {
    Card(Modifier.clearAndSetSemantics {
        contentDescription = "Analyzing. ${AnalysisSteps.steps[minOf(done, AnalysisSteps.steps.size - 1)].first}."
    }) {
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text("Analyzing", style = Type.rounded(Type.title3), color = Theme.ink)
            Text("On your phone. The photo never leaves it.", style = Type.subheadline, color = Theme.muted)
        }
        Column(Modifier.padding(top = 4.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            AnalysisSteps.steps.forEachIndexed { index, (title, detail) ->
                val isDone = index < done
                val isActive = index == done
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Box(Modifier.size(24.dp, 22.dp), contentAlignment = Alignment.Center) {
                        Crossfade(if (isDone) 2 else if (isActive) 1 else 0, label = "step") { phase ->
                            when (phase) {
                                2 -> Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = Theme.accent, modifier = Modifier.size(22.dp))
                                1 -> CircularProgressIndicator(Modifier.size(18.dp), color = Theme.muted, strokeWidth = 2.dp)
                                else -> Icon(Icons.Outlined.Circle, contentDescription = null, tint = Theme.track, modifier = Modifier.size(22.dp))
                            }
                        }
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            title,
                            style = Type.body.copy(fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal),
                            color = if (isDone || isActive) Theme.ink else Theme.muted,
                        )
                        AnimatedVisibility(isDone || isActive, enter = fadeIn() + expandVertically()) {
                            Text(detail, style = Type.footnote, color = Theme.muted)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AnswerCard(title: String, value: String, confidence: Double, modifier: Modifier, scale: @Composable () -> Unit) {
    Column(
        modifier
            .clip(RoundedCornerShape(20.dp))
            .background(Theme.surface)
            .padding(horizontal = 14.dp, vertical = 16.dp)
            .clearAndSetSemantics { contentDescription = "$title: $value, model confidence ${confidence.percent}" },
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Eyebrow(title)
        FitText(value, Type.rounded(Type.title2), Theme.ink, Modifier.heightIn(min = 56.dp))
        Spacer(Modifier.weight(1f, fill = false))
        scale()
        Divider()
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Confidence", Modifier.weight(1f), style = Type.caption, color = Theme.muted)
            Text(confidence.percent, style = Type.subheadline.copy(fontWeight = FontWeight.Bold), color = Theme.accent)
        }
        ConfidenceBar(confidence)
    }
}

/** Segmented scale with the predicted step highlighted. */
@Composable
private fun SegmentScale(labels: List<String>, colors: List<Color>, selected: Int, ramp: Boolean = false) {
    // With `ramp`, every step keeps its own colour and the selected one is outlined,
    // instead of only the selected step being filled.
    val track = Theme.track
    val accent = Theme.accent
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(Modifier.padding(horizontal = 3.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            labels.indices.forEach { index ->
                val fill = if (ramp) colors[index] else if (index == selected) colors[index] else track
                Box(
                    Modifier
                        .weight(1f)
                        .then(if (index == selected && ramp) Modifier.border(1.5.dp, accent, CircleShape).padding(3.dp) else Modifier.padding(3.dp))
                        .height(6.dp)
                        .clip(CircleShape)
                        .background(fill),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            labels.forEachIndexed { index, label ->
                Text(
                    label,
                    Modifier.weight(1f),
                    style = Type.caption2.copy(fontWeight = if (index == selected) FontWeight.Bold else FontWeight.Normal),
                    color = if (index == selected) Theme.ink else Theme.muted,
                    textAlign = when (index) {
                        0 -> TextAlign.Start
                        labels.size - 1 -> TextAlign.End
                        else -> TextAlign.Center
                    },
                    maxLines = 1,
                )
            }
        }
    }
}

@Composable
private fun ScoreTable(title: String, task: TaskPrediction, name: (String) -> String) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Eyebrow(title)
        task.ranked.forEach { (label, score) ->
            Row(
                Modifier.clearAndSetSemantics { contentDescription = "${name(label)}: ${score.percent}" },
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(name(label), Modifier.weight(1.3f), style = Type.subheadline, color = Theme.ink)
                ConfidenceBar(score, Modifier.weight(1f), tint = if (label == task.label) Theme.accent else Theme.muted.copy(alpha = 0.6f))
                Text(String.format(Locale.US, "%.2f", score), Modifier.width(36.dp), style = Type.subheadline, color = Theme.ink, textAlign = TextAlign.End)
            }
        }
    }
}

@Composable
private fun ScanNotesCard(scan: Scan) {
    Card {
        Text("Your notes", style = Type.rounded(Type.title3), color = Theme.ink)
        LabeledContent("Found", formatDate(scan.foundOn))
        scan.bodyArea?.let { LabeledContent("Bite spot", it.title) }
        if (scan.place.isNotEmpty()) LabeledContent("Where", scan.place)
        if (scan.notes.isNotEmpty()) Text(scan.notes, style = Type.subheadline, color = Theme.body)
    }
}

@Composable
private fun LabeledContent(label: String, value: String) {
    Row(Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}) {
        Text(label, Modifier.weight(1f), style = Type.subheadline, color = Theme.ink)
        Text(value, style = Type.subheadline, color = Theme.muted, textAlign = TextAlign.End)
    }
}
