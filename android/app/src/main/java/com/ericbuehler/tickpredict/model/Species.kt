package com.ericbuehler.tickpredict.model

import com.ericbuehler.tickpredict.inference.TickPrediction

/**
 * The eight species the v2 model knows, with what each can spread to people.
 * Diseases follow CDC's "Where ticks live" page (checked 2026-10-19).
 */
class Species private constructor(
    val label: String,
    val name: String,
    val otherName: String?,
    val carries: List<String>,
    val note: String?,
    val spreadsLyme: Boolean,
) {
    /** "Lyme disease, anaplasmosis, and babesiosis" style list. */
    val carriesText: String
        get() = when (carries.size) {
            0 -> ""
            1 -> carries[0]
            2 -> "${carries[0]} and ${carries[1]}"
            else -> carries.dropLast(1).joinToString(", ") + ", and " + carries.last()
        }

    companion object {
        operator fun invoke(label: String): Species =
            all[label] ?: Species(label, label, null, emptyList(), null, false)

        val lymeLabels = setOf("Ixodes scapularis", "Ixodes pacificus")

        private val all: Map<String, Species> = listOf(
            Species("Ixodes scapularis", "Blacklegged tick", "Deer tick",
                listOf("Lyme disease", "anaplasmosis", "babesiosis", "ehrlichiosis", "Powassan virus",
                    "hard tick relapsing fever"),
                null, true),
            Species("Ixodes pacificus", "Western blacklegged tick", null,
                listOf("Lyme disease", "anaplasmosis", "hard tick relapsing fever"),
                "Found on the Pacific coast. It looks almost the same as the eastern blacklegged (deer) tick.",
                true),
            Species("Dermacentor variabilis", "American dog tick", "Wood tick",
                listOf("Rocky Mountain spotted fever", "tularemia"),
                null, false),
            Species("Amblyomma americanum", "Lone star tick", null,
                listOf("ehrlichiosis", "tularemia", "Heartland virus", "Bourbon virus"),
                "Its bites are also linked to alpha-gal syndrome, an allergy to red meat, and to STARI, a rash like Lyme's.",
                false),
            Species("Amblyomma maculatum", "Gulf Coast tick", null,
                listOf("Rickettsia parkeri rickettsiosis, a kind of spotted fever"),
                null, false),
            Species("Rhipicephalus sanguineus", "Brown dog tick", null,
                listOf("Rocky Mountain spotted fever"),
                "Mostly a risk in the Southwest and along the U.S.–Mexico border. It often lives indoors with dogs.",
                false),
            Species("Dermacentor andersoni", "Rocky Mountain wood tick", null,
                listOf("Rocky Mountain spotted fever", "Colorado tick fever", "tularemia"),
                "Found in the Rocky Mountain states. It looks a lot like the American dog tick.",
                false),
            Species("Haemaphysalis longicornis", "Asian longhorned tick", null,
                emptyList(),
                "Not yet known to make people sick in the U.S. It spreads disease in other countries and can harm livestock.",
                false),
        ).associateBy { it.label }
    }
}

/**
 * How the species, life stage and engorgement combine for Lyme disease.
 * Doctors weigh a single preventive antibiotic dose when a blacklegged tick was
 * attached about 36 hours or more and it's within 72 hours of removal (IDSA 2020).
 */
enum class LymeRisk {
    HIGHER, POSSIBLE, LOWER, MALE_TICK, NOT_LYME_TICK;

    val title: String
        get() = when (this) {
            HIGHER -> "Higher Lyme risk"
            POSSIBLE -> "Possible Lyme risk"
            LOWER -> "Lower Lyme risk"
            MALE_TICK -> "Low Lyme risk"
            NOT_LYME_TICK -> "Doesn't spread Lyme"
        }

    val isUrgent: Boolean get() = this == HIGHER || this == POSSIBLE

    val explanation: String
        get() = when (this) {
            HIGHER -> "A fully engorged blacklegged tick was likely attached 36 hours or more, long enough to spread Lyme. If it came off within the last 72 hours, a doctor may offer a single preventive antibiotic dose. Call a doctor today."
            POSSIBLE -> "A semi-engorged blacklegged tick may have been attached long enough to spread Lyme. If it came off within the last 72 hours, ask a doctor whether a preventive antibiotic dose makes sense."
            LOWER -> "An unengorged tick likely wasn't attached long, and Lyme usually takes 36 hours or more to spread. Watch the bite for 30 days."
            MALE_TICK -> "Adult male blacklegged ticks rarely feed long enough to spread Lyme. Watch the bite for 30 days."
            NOT_LYME_TICK -> "This kind of tick isn't known to spread Lyme disease. Still watch the bite for 30 days."
        }

    companion object {
        /** Chance the model gives to either blacklegged species. */
        fun blackleggedScore(prediction: TickPrediction): Double? =
            prediction.species?.let { species -> Species.lymeLabels.sumOf { species.scores[it] ?: 0.0 } }

        /**
         * Engorged blacklegged ticks are under-represented in the species training
         * photos: on Penn's development photos (all blacklegged), only 6 of 16 fully
         * engorged ones were named correctly. So an engorged tick is never ruled out.
         */
        fun isEngorged(prediction: TickPrediction): Boolean = prediction.engorgement.label != "unengorged"

        /** Scans without a species have no Lyme reading. */
        fun of(prediction: TickPrediction): LymeRisk? {
            val blacklegged = blackleggedScore(prediction) ?: return null
            return when {
                blacklegged < 0.2 && !isEngorged(prediction) -> NOT_LYME_TICK
                prediction.lifeStage.label == "adult male" -> MALE_TICK
                else -> when (prediction.engorgement.label) {
                    "fully engorged" -> HIGHER
                    "semi-engorged" -> POSSIBLE
                    else -> LOWER
                }
            }
        }
    }
}
