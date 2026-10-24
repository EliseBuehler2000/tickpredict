package com.ericbuehler.tickpredict.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.ericbuehler.tickpredict.PendingPhoto
import com.ericbuehler.tickpredict.app
import java.util.UUID

enum class AppTab(val title: String) { SCAN("Scan"), HISTORY("History"), GUIDE("Guide") }

/** Screens pushed onto any tab's stack. Each keeps its own state while it's on the stack. */
sealed class Route {
    abstract val key: String

    class NewScan(val photo: PendingPhoto) : Route() {
        val state = ResultState()
        override val key = "new-${photo.id}"
    }

    class Saved(val scanId: String) : Route() {
        val state = ResultState().apply { savedScanId = scanId }
        override val key = "saved-$scanId-${UUID.randomUUID()}"
    }

    class FindCare : Route() {
        val finder = CareFinder()
        override val key = "care-${UUID.randomUUID()}"
    }
}

/** Tabs, each with its own stack, and the full-screen camera. */
class Navigator {
    var tab by mutableStateOf(AppTab.SCAN)
    var showCamera by mutableStateOf(false)
    val stacks: Map<AppTab, SnapshotStateList<Route>> = AppTab.entries.associateWith { mutableStateListOf() }

    val stack: SnapshotStateList<Route> get() = stacks.getValue(tab)

    fun push(route: Route) {
        stack.add(route)
    }

    fun pop() {
        stack.removeLastOrNull()
    }

    fun open(photo: PendingPhoto) {
        tab = AppTab.SCAN
        stacks.getValue(AppTab.SCAN).apply { clear(); add(Route.NewScan(photo)) }
    }

    fun scanAnother() {
        tab = AppTab.SCAN
        stacks.getValue(AppTab.SCAN).clear()
        showCamera = true
    }

    /** A scan deleted in one tab may still be open in another; close those screens. */
    fun dropDeletedScans(remaining: Set<String>) {
        for (stack in stacks.values) stack.removeAll { it is Route.Saved && it.scanId !in remaining }
    }
}

private data class Screen(val tab: AppTab, val route: Route?, val depth: Int)

@Composable
fun RootScreen(navigator: Navigator) {
    val context = LocalContext.current
    val store = context.app.store
    val holder = rememberSaveableStateHolder()

    LaunchedEffect(store) {
        store.scans.collect { scans -> navigator.dropDeletedScans(scans.map { it.id }.toSet()) }
    }

    BackHandler(enabled = navigator.stack.isNotEmpty() && !navigator.showCamera) { navigator.pop() }

    Box(Modifier.fillMaxSize().background(Theme.ground)) {
        Column(Modifier.fillMaxSize()) {
            val screen = Screen(navigator.tab, navigator.stack.lastOrNull(), navigator.stack.size)
            AnimatedContent(
                targetState = screen,
                modifier = Modifier.weight(1f),
                contentKey = { it.route?.key ?: "root-${it.tab}" },
                transitionSpec = {
                    when {
                        initialState.tab != targetState.tab -> fadeIn() togetherWith fadeOut()
                        targetState.depth > initialState.depth ->
                            slideInHorizontally { it } togetherWith slideOutHorizontally { -it / 4 } + fadeOut()
                        else -> slideInHorizontally { -it / 4 } + fadeIn() togetherWith slideOutHorizontally { it }
                    }
                },
                label = "screens",
            ) { shown ->
                holder.SaveableStateProvider(shown.route?.key ?: "root-${shown.tab}") {
                    Box(Modifier.fillMaxSize().background(Theme.ground)) {
                        ScreenContent(shown, navigator)
                    }
                }
            }
            TabBar(navigator)
        }

        AnimatedVisibility(
            visible = navigator.showCamera,
            enter = slideInVertically { it } + fadeIn(),
            exit = slideOutVertically { it } + fadeOut(),
        ) {
            BackHandler { navigator.showCamera = false }
            CaptureScreen(
                onClose = { navigator.showCamera = false },
                onCapture = { data ->
                    navigator.showCamera = false
                    navigator.open(PendingPhoto(data))
                },
            )
        }
    }
}

@Composable
private fun ScreenContent(screen: Screen, navigator: Navigator) {
    when (val route = screen.route) {
        null -> when (screen.tab) {
            AppTab.SCAN -> HomeScreen(
                openCamera = { navigator.showCamera = true },
                openHistory = { navigator.tab = AppTab.HISTORY },
                openGuide = { navigator.tab = AppTab.GUIDE },
                open = navigator::open,
                openScan = { navigator.push(Route.Saved(it)) },
            )
            AppTab.HISTORY -> HistoryScreen(openScan = { navigator.push(Route.Saved(it)) })
            AppTab.GUIDE -> GuideScreen(findCare = { navigator.push(Route.FindCare()) })
        }
        is Route.NewScan -> ResultScreen(
            source = ResultSource.New(route.photo),
            state = route.state,
            scanAnother = navigator::scanAnother,
            findCare = { navigator.push(Route.FindCare()) },
            onBack = navigator::pop,
        )
        is Route.Saved -> ResultScreen(
            source = ResultSource.Saved(route.scanId),
            state = route.state,
            scanAnother = navigator::scanAnother,
            findCare = { navigator.push(Route.FindCare()) },
            onBack = navigator::pop,
        )
        is Route.FindCare -> FindCareScreen(route.finder, onBack = navigator::pop)
    }
}

@Composable
private fun TabBar(navigator: Navigator) {
    NavigationBar(containerColor = Theme.surface, tonalElevation = 0.dp) {
        for (tab in AppTab.entries) {
            val selected = navigator.tab == tab
            NavigationBarItem(
                selected = selected,
                onClick = {
                    // Tapping the current tab returns to its first screen, as on iOS.
                    if (selected) navigator.stack.clear() else navigator.tab = tab
                },
                icon = {
                    Icon(
                        when (tab) {
                            AppTab.SCAN -> Icons.Filled.CenterFocusWeak
                            AppTab.HISTORY -> Icons.Filled.Schedule
                            AppTab.GUIDE -> Icons.AutoMirrored.Filled.MenuBook
                        },
                        contentDescription = null,
                    )
                },
                label = { Text(tab.title, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal) },
                modifier = Modifier.testTag("tab-${tab.title}"),
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = Theme.accent,
                    selectedTextColor = Theme.accent,
                    indicatorColor = Theme.accentTint,
                    unselectedIconColor = Theme.muted,
                    unselectedTextColor = Theme.muted,
                ),
            )
        }
    }
}
