package com.ericbuehler.tickpredict.model

import com.ericbuehler.tickpredict.inference.TickPrediction
import com.ericbuehler.tickpredict.ui.percent
import kotlinx.serialization.Serializable

/** A saved scan. Everything stays on the device; the photo and thumbnail are files next to the index. */
@Serializable
data class Scan(
    val id: String,
    /** When the photo was taken, epoch milliseconds. */
    val createdAt: Long,
    val prediction: TickPrediction,
    val foundOn: Long,
    val bodyArea: BodyArea? = null,
    val place: String = "",
    val notes: String = "",
) {
    val title: String get() = prediction.title

    val searchText: String
        get() = listOf(
            title, prediction.species?.let { Species(it.label).otherName ?: "" } ?: "",
            Engorgement(prediction.engorgement.label).name, place, notes, bodyArea?.title ?: "",
        ).joinToString(" ")
}

/** Where the bite was; the marker is on the body outline, in its 74 × 150 coordinate space. */
@Serializable
enum class BodyArea(val title: String, val markerX: Float, val markerY: Float) {
    HEAD("Head / neck", 37f, 18f),
    ARM("Arm", 15f, 52f),
    TORSO("Torso", 37f, 50f),
    GROIN("Groin", 37f, 78f),
    KNEE("Back of knee", 29f, 104f),
    LEG("Leg", 45f, 120f),
    FOOT("Ankle / foot", 29f, 138f),
}

/** "Blacklegged tick · Nymph · Semi-engorged"; scans without a species leave it out. */
val TickPrediction.title: String
    get() = listOfNotNull(species?.let { Species(it.label).name }, LifeStage(lifeStage.label).name,
        Engorgement(engorgement.label).name).joinToString(" · ")

val TickPrediction.confidenceSummary: String
    get() = listOfNotNull(species, lifeStage, engorgement).joinToString(" · ") { it.confidence.percent } +
        " model confidence"

/** Display names and plain-language explanations for each class. */
class LifeStage(val label: String) {
    val name: String
        get() = when (label) {
            "nymph" -> "Nymph"
            "adult male" -> "Adult male"
            else -> "Adult female"
        }

    val shortName: String
        get() = when (label) {
            "nymph" -> "Nymph"
            "adult male" -> "Male"
            else -> "Female"
        }

    val meaning: String
        get() = when (label) {
            "nymph" -> "Nymphs are the young stage, about the size of a poppy seed. They're easy to miss, so they often stay attached longer. In deer ticks, they cause most Lyme infections in people."
            "adult male" -> "Adult males are about the size of a sesame seed and stay small when they feed. Male blacklegged ticks rarely feed long at all."
            else -> "Adult females are about the size of a sesame seed before feeding and grow much larger as they feed. They're easier to spot than nymphs."
        }

    companion object {
        val order = listOf("nymph", "adult female", "adult male")
    }
}

class Engorgement(val label: String) {
    val index: Int get() = order.indexOf(label).coerceAtLeast(0)

    val name: String
        get() = when (label) {
            "unengorged" -> "Unengorged"
            "semi-engorged" -> "Semi-engorged"
            else -> "Fully engorged"
        }

    val shortName: String get() = listOf("None", "Semi", "Fully")[index]

    val meaning: String
        get() = when (label) {
            "unengorged" -> "Unengorged means it's still flat, with little or no blood taken, so it probably wasn't attached long. It can't tell you exactly how long."
            "semi-engorged" -> "Semi-engorged means it has taken some blood, so it was likely attached for a while. It can't tell you exactly how long."
            else -> "Fully engorged means it's swollen with blood, which usually takes days of feeding. Tell your doctor it looked engorged."
        }

    companion object {
        val order = listOf("unengorged", "semi-engorged", "fully engorged")
    }
}
