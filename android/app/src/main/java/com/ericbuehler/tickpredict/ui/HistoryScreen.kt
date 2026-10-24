package com.ericbuehler.tickpredict.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.SwipeToDismissBox
import androidx.compose.material3.SwipeToDismissBoxValue
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.rememberSwipeToDismissBoxState
import androidx.compose.runtime.Composable
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ericbuehler.tickpredict.app
import com.ericbuehler.tickpredict.model.Scan
import kotlinx.coroutines.launch
import java.text.Normalizer

@Composable
fun HistoryScreen(openScan: (String) -> Unit) {
    val context = LocalContext.current
    val store = context.app.store
    val scope = rememberCoroutineScope()
    val all by store.scans.collectAsStateWithLifecycle()
    var search by rememberSaveable { mutableStateOf("") }
    val scans = remember(all) { all.sortedWith(compareByDescending<Scan> { it.foundOn }.thenByDescending { it.createdAt }) }
    val query = search.trim()
    val filtered = if (query.isEmpty()) scans else scans.filter { matches(it.searchText, query) }
    // Grouped by when the tick was found, which is the date each row shows.
    val weekAgo = System.currentTimeMillis() - 7 * 86_400_000L
    val groups = listOf(
        "This week" to filtered.filter { it.foundOn >= weekAgo },
        "Earlier" to filtered.filter { it.foundOn < weekAgo },
    ).filter { it.second.isNotEmpty() }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(horizontal = 20.dp)) {
            LargeTitle("History")
            if (scans.isNotEmpty()) SearchField(search) { search = it }
        }
        when {
            scans.isEmpty() -> ContentUnavailable(
                Icons.Filled.Schedule, "No scans yet",
                "Scans you save appear here, with the date and where the bite was.",
            )
            filtered.isEmpty() -> ContentUnavailable(
                Icons.Filled.Search, "No Results for “$query”",
                "Check the spelling or try a new search.",
            )
            else -> LazyColumn(
                Modifier.fillMaxSize(),
                contentPadding = PaddingValues(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 24.dp),
            ) {
                for ((title, group) in groups) {
                    item(key = "header-$title") {
                        Eyebrow(title, Modifier.padding(start = 4.dp, top = 12.dp, bottom = 8.dp))
                    }
                    items(group, key = { it.id }) { scan ->
                        val first = scan == group.first()
                        val last = scan == group.last()
                        val shape = RoundedCornerShape(
                            topStart = if (first) 18.dp else 0.dp, topEnd = if (first) 18.dp else 0.dp,
                            bottomStart = if (last) 18.dp else 0.dp, bottomEnd = if (last) 18.dp else 0.dp,
                        )
                        DeletableRow(onDelete = { scope.launch { store.delete(scan.id) } }) {
                            Column(Modifier.clip(shape).background(Theme.surface)) {
                                ScanRow(scan, store.thumbnailFile(scan.id), Modifier.clickable { openScan(scan.id) }, showsChevron = false)
                                if (!last) Divider(Modifier.padding(start = 76.dp))
                            }
                        }
                    }
                }
            }
        }
    }
}

/** Case- and accent-insensitive, like `localizedStandardContains`. */
private fun matches(text: String, query: String): Boolean = fold(text).contains(fold(query))

private fun fold(text: String) =
    Normalizer.normalize(text, Normalizer.Form.NFD).replace(Regex("\\p{Mn}+"), "").lowercase()

@Composable
private fun SearchField(value: String, onChange: (String) -> Unit) {
    TextField(
        value = value,
        onValueChange = onChange,
        modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
        placeholder = { Text("Search notes or places", style = Type.body, color = Theme.muted) },
        leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null, tint = Theme.muted) },
        trailingIcon = {
            if (value.isNotEmpty()) {
                IconButton(onClick = { onChange("") }) { Icon(Icons.Filled.Close, contentDescription = "Clear search", tint = Theme.muted) }
            }
        },
        singleLine = true,
        textStyle = Type.body.copy(color = Theme.ink),
        shape = RoundedCornerShape(12.dp),
        colors = TextFieldDefaults.colors(
            focusedContainerColor = Theme.track,
            unfocusedContainerColor = Theme.track,
            focusedIndicatorColor = Color.Transparent,
            unfocusedIndicatorColor = Color.Transparent,
            cursorColor = Theme.accent,
        ),
    )
}

/** Swipe left to delete, as in an iOS list. */
@Composable
private fun DeletableRow(onDelete: () -> Unit, content: @Composable () -> Unit) {
    val state = rememberSwipeToDismissBoxState(confirmValueChange = { value ->
        if (value == SwipeToDismissBoxValue.EndToStart) onDelete()
        value == SwipeToDismissBoxValue.EndToStart
    })
    SwipeToDismissBox(
        state = state,
        enableDismissFromStartToEnd = false,
        backgroundContent = {
            // Only while swiping, so the red doesn't show around the row's rounded corners.
            if (state.dismissDirection != SwipeToDismissBoxValue.EndToStart) return@SwipeToDismissBox
            Box(
                Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(18.dp))
                    .background(Theme.bite)
                    .padding(horizontal = 20.dp),
                contentAlignment = Alignment.CenterEnd,
            ) {
                Icon(Icons.Filled.Delete, contentDescription = "Delete", tint = Color.White)
            }
        },
    ) { content() }
}
