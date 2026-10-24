package com.ericbuehler.tickpredict.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.ericbuehler.tickpredict.model.Engorgement

@Composable
fun GuideScreen(findCare: () -> Unit) {
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
            .padding(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        LargeTitle("Tick guide")

        Card {
            Text("Remove it safely", style = Type.rounded(Type.title3), color = Theme.ink)
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                NumberedLine(1, "Grip the tick with fine-tipped tweezers, as close to the skin as you can.")
                NumberedLine(2, "Pull straight up with steady pressure. Don't twist or jerk.")
                NumberedLine(3, "Clean the bite and your hands with soap or rubbing alcohol.")
                NumberedLine(4, "Keep the tick for scanning. Never crush it with your fingers.")
            }
            Text("Skip nail polish, petroleum jelly, or heat. Get it out quickly instead.", style = Type.subheadline, color = Theme.muted)
        }

        Card {
            Text("Nymph or adult?", style = Type.rounded(Type.title3), color = Theme.ink)
            val dark = Theme.isDark
            Row(Modifier.height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SizeTile("Nymph", "Poppy seed size", Modifier.weight(1f)) {
                    Canvas(Modifier.size(8.dp)) { drawCircle(if (dark) Color(0xFFC9B6A6) else Color(0xFF3B2A20)) }
                }
                SizeTile("Adult female", "Sesame seed size, unfed", Modifier.weight(1f)) {
                    Canvas(Modifier.size(16.dp, 22.dp)) {
                        drawOval(if (dark) Color(0xFFD9876B) else Color(0xFF8E3B26))
                        val head = Size(12.dp.toPx(), 9.dp.toPx())
                        drawOval(if (dark) Color(0xFF8A7466) else Color(0xFF2B1E16), Offset((size.width - head.width) / 2, 0f), head)
                    }
                }
                SizeTile("Adult male", "Sesame seed size, stays flat", Modifier.weight(1f)) {
                    Canvas(Modifier.size(14.dp, 19.dp)) { drawOval(if (dark) Color(0xFF8A7466) else Color(0xFF2B1E16)) }
                }
            }
            Text(
                "Nymphs are tiny and easy to miss, so they often stay attached longer. In deer ticks, they cause most Lyme infections in people. Males rarely feed long.",
                style = Type.subheadline, color = Theme.body,
            )
        }

        Card {
            Text("What engorgement means", style = Type.rounded(Type.title3), color = Theme.ink)
            val ramp = Theme.engorgementRamp
            Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Engorgement.order.forEachIndexed { index, label ->
                    Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Canvas(Modifier.size(Dp(18f + index * 12), Dp(24f + index * 12))) { drawOval(ramp[index]) }
                        val short = Engorgement(label).shortName
                        Text(if (short == "None") "Unengorged" else short, style = Type.footnote.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink)
                        Text(
                            listOf("Flat, little or no blood", "Rounder, partly fed", "Swollen, fed for days")[index],
                            style = Type.caption, color = Theme.muted, textAlign = TextAlign.Center,
                        )
                    }
                }
            }
            Text(
                "The longer a tick feeds, the more it swells, and the more chance it has to pass on an infection. Tell your doctor if yours looked engorged.",
                style = Type.subheadline, color = Theme.body,
            )
        }

        Card(fill = Theme.accentTint) {
            Text("When to call a doctor", style = Type.rounded(Type.title3), color = Theme.ink)
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Bullet("A rash, especially one that spreads or looks like a bull's-eye")
                Bullet("Fever, chills, headache, tiredness, or achy joints within 30 days")
                Bullet("You couldn't remove the whole tick")
                Bullet("The tick was engorged or attached for a long time")
            }
            FindCareButton(findCare)
        }

        Card {
            Text("About TickPredict", style = Type.rounded(Type.title3), color = Theme.ink)
            val style = Type.subheadline
            Text("Two image models, trained on about 13,000 iNaturalist phone photos and Penn research photos. They know 8 common U.S. ticks: blacklegged (deer), western blacklegged, American dog, lone star, Gulf Coast, brown dog, Rocky Mountain wood, and Asian longhorned.", style = style, color = Theme.body)
            Text("On photos held out from training, they told blacklegged ticks from others 97% of the time, named the exact species 85%, life stage 93–96%, and engorgement 86%.", style = style, color = Theme.body)
            Text("They can't tell larvae or other kinds of tick, and confidence is the model's own score, not a guarantee.", style = style, color = Theme.body)
            Text("Everything runs on your phone. Photos never leave it.", style = style, color = Theme.body)
            Text("Models v2.0 · engorgement v1.0", style = Type.footnote, color = Theme.muted)
        }
    }
}

@Composable
private fun SizeTile(name: String, detail: String, modifier: Modifier, mark: @Composable () -> Unit) {
    Column(
        modifier
            .fillMaxHeight()
            .clip(RoundedCornerShape(14.dp))
            .background(Theme.ground)
            .padding(14.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(Modifier.height(40.dp), contentAlignment = Alignment.Center) { mark() }
        Text(name, style = Type.subheadline.copy(fontWeight = FontWeight.SemiBold), color = Theme.ink, textAlign = TextAlign.Center)
        Text(detail, style = Type.footnote, color = Theme.muted, textAlign = TextAlign.Center)
    }
}

@Composable
private fun Bullet(text: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("•", style = Type.subheadline, color = Theme.accent)
        Text(text, style = Type.subheadline, color = Theme.body)
    }
}
