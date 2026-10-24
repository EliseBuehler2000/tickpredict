package com.ericbuehler.tickpredict.ui

import android.graphics.BitmapFactory
import android.view.HapticFeedbackConstants
import androidx.compose.animation.core.animateOffsetAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.ericbuehler.tickpredict.model.Scan
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

/** The filled accent button (iOS `.glassProminent`). */
@Composable
fun ProminentButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    Button(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        enabled = enabled,
        shape = CircleShape,
        colors = ButtonDefaults.buttonColors(containerColor = Theme.accent, contentColor = Theme.onAccent),
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 10.dp),
        content = content,
    )
}

/** The quieter button beside a prominent one (iOS `.glass`). */
@Composable
fun SecondaryButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable RowScope.() -> Unit,
) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        shape = CircleShape,
        border = BorderStroke(1.dp, Theme.line),
        colors = ButtonDefaults.outlinedButtonColors(containerColor = Theme.surface, contentColor = Theme.ink),
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 10.dp),
        content = content,
    )
}

/** A text button in the accent colour, like an iOS plain button. */
@Composable
fun LinkButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, style: TextStyle = Type.subheadline) {
    Text(
        text,
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .clickable(role = Role.Button, onClick = onClick)
            .padding(vertical = 6.dp),
        style = style,
        color = Theme.accent,
    )
}

@Composable
fun IconLabel(icon: ImageVector, text: String, modifier: Modifier = Modifier) {
    Row(modifier, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(20.dp))
        Text(text, style = Type.headline)
    }
}

/** The big rounded title at the top of each tab, like an iOS large title. */
@Composable
fun LargeTitle(text: String, modifier: Modifier = Modifier) {
    Text(
        text,
        modifier = modifier
            .statusBarsPadding()
            .padding(top = 12.dp, bottom = 4.dp)
            .semantics { heading() },
        style = Type.largeTitle,
        color = Theme.ink,
    )
}

/** An inline title bar for pushed screens, with a back button and trailing actions. */
@Composable
fun InlineTopBar(title: String, onBack: () -> Unit, actions: @Composable RowScope.() -> Unit = {}) {
    Box(
        Modifier
            .fillMaxWidth()
            .background(Theme.ground)
            .statusBarsPadding()
            .heightIn(min = 52.dp)
            .padding(horizontal = 4.dp),
    ) {
        IconButton(onClick = onBack, modifier = Modifier.align(Alignment.CenterStart)) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Theme.accent)
        }
        Text(
            title,
            modifier = Modifier.align(Alignment.Center).semantics { heading() },
            style = Type.headline,
            color = Theme.ink,
        )
        Row(Modifier.align(Alignment.CenterEnd), verticalAlignment = Alignment.CenterVertically, content = actions)
    }
}

/** Centered icon, title and message for empty screens (iOS `ContentUnavailableView`). */
@Composable
fun ContentUnavailable(icon: ImageVector, title: String, description: String, modifier: Modifier = Modifier) {
    Column(
        modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(icon, contentDescription = null, tint = Theme.muted, modifier = Modifier.size(48.dp))
        Text(title, style = Type.rounded(Type.title2), color = Theme.ink, textAlign = TextAlign.Center)
        Text(description, style = Type.subheadline, color = Theme.muted, textAlign = TextAlign.Center)
    }
}

@Composable
fun NumberedLine(number: Int, text: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("$number.", style = Type.subheadline.copy(fontWeight = FontWeight.SemiBold), color = Theme.accent)
        Text(text, style = Type.subheadline, color = Theme.body)
    }
}

/** One saved scan in a list: thumbnail, result, and when/where. */
@Composable
fun ScanRow(scan: Scan, thumbnail: File, modifier: Modifier = Modifier, showsChevron: Boolean = true) {
    Row(
        modifier
            .fillMaxWidth()
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        ScanThumbnail(thumbnail, 50.dp)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(scan.title, style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink)
            Text(scanSubtitle(scan), style = Type.footnote, color = Theme.muted)
            if (scan.notes.isNotEmpty()) {
                Text(scan.notes, style = Type.footnote, color = Theme.body, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
        if (showsChevron) {
            Icon(Icons.Filled.ChevronRight, contentDescription = null, tint = Theme.muted.copy(alpha = 0.6f))
        }
    }
}

private fun scanSubtitle(scan: Scan): String {
    val found = Instant.ofEpochMilli(scan.foundOn).atZone(ZoneId.systemDefault()).toLocalDate()
    val date = if (found == LocalDate.now()) "Today" else found.format(DateTimeFormatter.ofPattern("MMM d"))
    return listOfNotNull(date, scan.bodyArea?.title, scan.place.ifEmpty { null }).joinToString(" · ")
}

@Composable
fun ScanThumbnail(file: File?, size: Dp) {
    val image by produceState<ImageBitmap?>(null, file) {
        value = withContext(Dispatchers.IO) {
            file?.takeIf { it.exists() }?.let { BitmapFactory.decodeFile(it.path)?.asImageBitmap() }
        }
    }
    ThumbnailFrame(image, size)
}

@Composable
fun ThumbnailFrame(image: ImageBitmap?, size: Dp) {
    Box(
        Modifier
            .size(size)
            .clip(RoundedCornerShape(12.dp))
            .background(Theme.photoBackdrop)
            .clearAndSetSemantics {},
        contentAlignment = Alignment.Center,
    ) {
        if (image != null) {
            Image(image, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        } else {
            Icon(Icons.Filled.BugReport, contentDescription = null, tint = Theme.muted)
        }
    }
}

@Composable
fun StepRow(icon: ImageVector, title: String, text: String) {
    Row(
        Modifier.semantics(mergeDescendants = true) {},
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Icon(icon, contentDescription = null, tint = Theme.accent, modifier = Modifier.width(26.dp).size(22.dp))
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = Type.body.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink)
            Text(text, style = Type.subheadline, color = Theme.body)
        }
    }
}

@Composable
fun Chip(title: String, selected: Boolean, onClick: () -> Unit) {
    Text(
        title,
        modifier = Modifier
            .heightIn(min = 36.dp)
            .clip(CircleShape)
            .background(if (selected) Theme.accentTint else Theme.surface)
            .border(1.5.dp, if (selected) Theme.accent else Theme.line, CircleShape)
            .clickable(role = Role.Checkbox, onClick = onClick)
            .semantics { this.selected = selected }
            .padding(horizontal = 12.dp, vertical = 8.dp),
        style = Type.subheadline.copy(fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal),
        color = if (selected) Theme.accent else Theme.body,
    )
}

/** Simple front-facing figure with the bite spot marked, in a 74 × 150 space. */
@Composable
fun BodyOutline(marker: Offset?) {
    val shade = Theme.track
    val bite = Theme.bite
    val ring = Theme.surface
    val animated by animateOffsetAsState(marker ?: Offset.Zero, label = "marker")
    Canvas(Modifier.size(74.dp, 150.dp).clearAndSetSemantics {}) {
        val unit = size.width / 74f
        fun rect(x: Float, y: Float, w: Float, h: Float) {
            drawRoundRect(shade, Offset(x * unit, y * unit), Size(w * unit, h * unit),
                CornerRadius(minOf(w, h) / 2 * unit))
        }
        drawCircle(shade, 11 * unit, Offset(37 * unit, 14 * unit))
        rect(22f, 28f, 30f, 50f)
        rect(10f, 30f, 10f, 46f)
        rect(54f, 30f, 10f, 46f)
        rect(23f, 74f, 12f, 70f)
        rect(39f, 74f, 12f, 70f)
        if (marker != null) {
            val center = Offset(animated.x * unit, animated.y * unit)
            drawCircle(bite, 7 * unit, center)
            drawCircle(ring, 7 * unit, center, style = Stroke(2 * unit))
        }
    }
}

/** Success and selection haptics, like iOS `sensoryFeedback`. */
class Haptics(private val view: android.view.View) {
    fun success() {
        view.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
    }

    fun selection() {
        view.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
    }

    fun impact() {
        view.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
    }
}

@Composable
fun rememberHaptics(): Haptics {
    val view = LocalView.current
    return remember(view) { Haptics(view) }
}

@Composable
fun Divider(modifier: Modifier = Modifier) {
    Box(
        modifier
            .fillMaxWidth()
            .heightIn(min = 1.dp, max = 1.dp)
            .background(Theme.line),
    )
}

fun formatDate(millis: Long, style: FormatStyle = FormatStyle.MEDIUM): String =
    Instant.ofEpochMilli(millis).atZone(ZoneId.systemDefault()).toLocalDate()
        .format(DateTimeFormatter.ofLocalizedDate(style))

fun formatDateTime(millis: Long): String {
    val time = Instant.ofEpochMilli(millis).atZone(ZoneId.systemDefault())
    val clock = time.format(DateTimeFormatter.ofLocalizedTime(FormatStyle.SHORT))
    return if (time.toLocalDate() == LocalDate.now()) "Today, $clock"
    else time.format(DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM, FormatStyle.SHORT))
}

/**
 * Text that shrinks (down to `minScale`) until no word is broken across lines, like
 * iOS `minimumScaleFactor`, so a long word such as "Semi-engorged" stays whole.
 */
@Composable
fun FitText(text: String, style: TextStyle, color: Color, modifier: Modifier = Modifier, minScale: Float = 0.75f) {
    var scale by remember(text) { mutableFloatStateOf(1f) }
    var settled by remember(text) { mutableStateOf(false) }
    Text(
        text,
        modifier = modifier.drawWithContent { if (settled) drawContent() },
        style = style.copy(fontSize = style.fontSize * scale, lineHeight = style.lineHeight * scale),
        color = color,
        onTextLayout = { layout ->
            val broken = (0 until layout.lineCount - 1).any { line ->
                val end = layout.getLineEnd(line)
                end in 1 until text.length && text[end - 1].isLetter() && text[end].isLetter()
            }
            if (broken && scale > minScale) scale -= 0.05f else settled = true
        },
    )
}
