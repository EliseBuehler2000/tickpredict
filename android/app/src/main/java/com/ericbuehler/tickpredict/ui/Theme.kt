package com.ericbuehler.tickpredict.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ericbuehler.tickpredict.R
import java.text.NumberFormat

/** The "calm natural" palette from the design mockup, with dark-mode pairs. */
@Immutable
data class Palette(
    val ground: Color,
    val surface: Color,
    val ink: Color,
    val body: Color,
    val muted: Color,
    val accent: Color,
    val onAccent: Color,
    val accentTint: Color,
    val line: Color,
    val track: Color,
    val photoBackdrop: Color,
    val warningFill: Color,
    val warningInk: Color,
    val bite: Color,
    /** Unengorged -> semi -> fully, increasing in depth. */
    val engorgementRamp: List<Color>,
    /** The green "Scan a tick" card. */
    val scanCard: Color,
    val isDark: Boolean,
)

private fun hex(value: Long) = Color(0xFF000000 or value)

private val LightPalette = Palette(
    ground = hex(0xF6F4EE), surface = hex(0xFFFFFF), ink = hex(0x18241C), body = hex(0x2A3A2F),
    muted = hex(0x56605A), accent = hex(0x2E5A3A), onAccent = hex(0xFFFFFF), accentTint = hex(0xE3EBDD),
    line = hex(0xECE9E1), track = hex(0xE9E6DD), photoBackdrop = hex(0xEFEBE2), warningFill = hex(0xFBEDE6),
    warningInk = hex(0x6B2410), bite = hex(0xC2410C),
    engorgementRamp = listOf(hex(0xEFE4D3), hex(0xC98B5A), hex(0x6E3A22)),
    scanCard = hex(0x2E5A3A), isDark = false,
)

private val DarkPalette = Palette(
    ground = hex(0x111512), surface = hex(0x1B201C), ink = hex(0xEDF1EC), body = hex(0xD3DAD4),
    muted = hex(0xA0A9A3), accent = hex(0x86BC93), onAccent = hex(0x0F1F14), accentTint = hex(0x1E2C22),
    line = hex(0x2A312C), track = hex(0x2D352F), photoBackdrop = hex(0x262B27), warningFill = hex(0x3A231B),
    warningInk = hex(0xF4BFA6), bite = hex(0xF97B4A),
    engorgementRamp = listOf(hex(0x5B4C3D), hex(0xB57A4C), hex(0xE7B58D)),
    scanCard = hex(0x24472E), isDark = true,
)

private val LocalPalette = staticCompositionLocalOf { LightPalette }

/** Colours from the current palette, named as in the iOS app. */
object Theme {
    val ground: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.ground
    val surface: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.surface
    val ink: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.ink
    val body: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.body
    val muted: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.muted
    val accent: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.accent
    val onAccent: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.onAccent
    val accentTint: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.accentTint
    val line: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.line
    val track: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.track
    val photoBackdrop: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.photoBackdrop
    val warningFill: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.warningFill
    val warningInk: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.warningInk
    val bite: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.bite
    val engorgementRamp: List<Color> @Composable @ReadOnlyComposable get() = LocalPalette.current.engorgementRamp
    val scanCard: Color @Composable @ReadOnlyComposable get() = LocalPalette.current.scanCard
    val isDark: Boolean @Composable @ReadOnlyComposable get() = LocalPalette.current.isDark
}

/** Rounded bold headings, standing in for SF Rounded. */
@OptIn(ExperimentalTextApi::class)
val Rounded = FontFamily(
    Font(R.font.nunito, FontWeight.Bold, variationSettings = FontVariation.Settings(FontVariation.weight(750))),
    Font(R.font.nunito, FontWeight.SemiBold, variationSettings = FontVariation.Settings(FontVariation.weight(650))),
)

/** iOS text styles at their default sizes. */
object Type {
    val largeTitle = TextStyle(fontFamily = Rounded, fontWeight = FontWeight.Bold, fontSize = 34.sp, lineHeight = 41.sp)
    val title = TextStyle(fontSize = 28.sp, lineHeight = 34.sp)
    val title2 = TextStyle(fontSize = 22.sp, lineHeight = 28.sp)
    val title3 = TextStyle(fontSize = 20.sp, lineHeight = 25.sp)
    val headline = TextStyle(fontSize = 17.sp, lineHeight = 22.sp, fontWeight = FontWeight.SemiBold)
    val body = TextStyle(fontSize = 17.sp, lineHeight = 22.sp)
    val subheadline = TextStyle(fontSize = 15.sp, lineHeight = 20.sp)
    val footnote = TextStyle(fontSize = 13.sp, lineHeight = 18.sp)
    val caption = TextStyle(fontSize = 12.sp, lineHeight = 16.sp)
    val caption2 = TextStyle(fontSize = 11.sp, lineHeight = 13.sp)

    /** `.font(.rounded(style))`: the style in bold rounded type. */
    fun rounded(style: TextStyle) = style.copy(fontFamily = Rounded, fontWeight = FontWeight.Bold)
}

@Composable
fun TickPredictTheme(content: @Composable () -> Unit) {
    val palette = if (isSystemInDarkTheme()) DarkPalette else LightPalette
    val scheme = (if (palette.isDark) darkColorScheme() else lightColorScheme()).copy(
        primary = palette.accent,
        onPrimary = palette.onAccent,
        primaryContainer = palette.accentTint,
        onPrimaryContainer = palette.ink,
        secondaryContainer = palette.accentTint,
        onSecondaryContainer = palette.ink,
        background = palette.ground,
        onBackground = palette.ink,
        surface = palette.ground,
        onSurface = palette.ink,
        onSurfaceVariant = palette.muted,
        surfaceContainer = palette.surface,
        surfaceContainerLow = palette.surface,
        surfaceContainerHigh = palette.surface,
        surfaceContainerHighest = palette.track,
        surfaceContainerLowest = palette.surface,
        outline = palette.line,
        outlineVariant = palette.line,
        error = palette.bite,
    )
    CompositionLocalProvider(LocalPalette provides palette) {
        MaterialTheme(colorScheme = scheme, content = content)
    }
}

/** A rounded content card on the warm background. */
@Composable
fun Card(
    modifier: Modifier = Modifier,
    fill: Color = Theme.surface,
    padding: Dp = 18.dp,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(fill)
            .padding(padding),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        content = content,
    )
}

/** Small spaced capitals used above each value. */
@Composable
fun Eyebrow(text: String, modifier: Modifier = Modifier) {
    Text(
        text.uppercase(),
        modifier = modifier,
        style = Type.caption.copy(fontWeight = FontWeight.SemiBold, letterSpacing = 0.6.sp),
        color = Theme.muted,
    )
}

@Composable
fun ConfidenceBar(value: Double, modifier: Modifier = Modifier, height: Dp = 6.dp, tint: Color = Theme.accent) {
    Box(
        modifier
            .fillMaxWidth()
            .height(height)
            .clip(CircleShape)
            .background(Theme.track),
    ) {
        Box(
            Modifier
                .fillMaxHeight()
                .fillMaxWidth(value.coerceIn(0.0, 1.0).toFloat())
                .clip(CircleShape)
                .background(tint),
        )
    }
}

/** "86%", rounding halves to even like iOS. */
val Double.percent: String
    get() = NumberFormat.getPercentInstance().apply { maximumFractionDigits = 0 }.format(this)
