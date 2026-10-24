package com.ericbuehler.tickpredict

import android.content.res.Configuration
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import com.ericbuehler.tickpredict.ui.Navigator
import com.ericbuehler.tickpredict.ui.RootScreen
import com.ericbuehler.tickpredict.ui.TickPredictTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val launch = if (BuildConfig.DEBUG && savedInstanceState == null) DebugLaunch.from(intent) else null
        setContent {
            TickPredictTheme {
                val navigator = remember { Navigator() }
                LaunchedEffect(Unit) { launch?.apply(this@MainActivity, navigator) }
                RootScreen(navigator)
            }
        }
    }

    /** Dark mode changes don't recreate the activity (see the manifest), so refresh the bar icons. */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        enableEdgeToEdge()
    }
}
