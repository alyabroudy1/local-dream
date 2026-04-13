package io.github.xororz.localdream.ui.screens

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.Log
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.activity.compose.BackHandler
import androidx.compose.ui.res.stringResource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.Base64
import kotlin.math.max
import kotlin.math.sqrt
import android.content.Context
import androidx.compose.ui.graphics.Color as ComposeColor
import io.github.xororz.localdream.R
import io.github.xororz.localdream.ml.MobileSAMSegmenter
import io.github.xororz.localdream.ml.SegmentedObject
import androidx.core.content.edit
import androidx.compose.ui.draw.clipToBounds
import androidx.core.graphics.createBitmap
import io.github.xororz.localdream.ml.SmartPromptProcessor
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import kotlin.math.min

enum class ToolMode {
    BRUSH,
    ERASER,
    SMART_SELECT,
    AUTO_DETECT
}

data class PathData(
    val points: List<Offset>,
    val size: Float,
    val mode: ToolMode,
    val color: Int = Color.WHITE
)

private data class FacialMaskRect(
    val centerXPct: Float, val centerYPct: Float,
    val widthPct: Float, val heightPct: Float
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InpaintScreen(
    originalBitmap: Bitmap,
    existingMaskBitmap: Bitmap? = null,
    existingPathHistory: List<PathData>? = null,
    onInpaintComplete: (String, Bitmap, Bitmap, List<PathData>) -> Unit,
    onSmartEdit: ((maskBase64: String, prompt: String, negPrompt: String, denoise: Float, cfg: Float) -> Unit)? = null,
    onCancel: () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val context = LocalContext.current
    val density = LocalDensity.current

    val sharedPrefs =
        remember { context.getSharedPreferences("inpaint_prefs", Context.MODE_PRIVATE) }
    val defaultColor = Color.WHITE
    val savedColor = remember { sharedPrefs.getInt("brush_color", defaultColor) }
    val savedToolMode =
        remember { sharedPrefs.getString("tool_mode", ToolMode.BRUSH.name) ?: ToolMode.BRUSH.name }

    var brushColor by remember { mutableStateOf(savedColor) }
    var showColorPicker by remember { mutableStateOf(false) }
    var currentToolMode by remember { mutableStateOf(
        try { ToolMode.valueOf(savedToolMode) } catch (_: Exception) { ToolMode.BRUSH }
    ) }

    // Smart Select (MobileSAM) state
    val samSegmenter = remember { MobileSAMSegmenter(context) }
    var isSamModelLoaded by remember { mutableStateOf(false) }
    var isSamProcessing by remember { mutableStateOf(false) }
    var samLoadingMessage by remember { mutableStateOf("") }

    // Auto Detect state
    var isScanning by remember { mutableStateOf(false) }
    var scanProgress by remember { mutableStateOf("") }
    var detectedObjects by remember { mutableStateOf<List<SegmentedObject>>(emptyList()) }
    val selectedObjectIds = remember { mutableStateListOf<Int>() }
    var smartEditText by remember { mutableStateOf("") }
    var processedPrompt by remember { mutableStateOf<SmartPromptProcessor.ProcessedPrompt?>(null) }

    // Load SAM model in background on first compose
    LaunchedEffect(Unit) {
        isSamModelLoaded = samSegmenter.loadModel()
        if (isSamModelLoaded) {
            Log.i("InpaintScreen", "MobileSAM model loaded")
        } else {
            Log.w("InpaintScreen", "MobileSAM model failed to load")
        }
    }

    // Clean up SAM cache on dispose
    DisposableEffect(Unit) {
        onDispose {
            samSegmenter.clearCache()
        }
    }

    LaunchedEffect(currentToolMode) {
        sharedPrefs.edit { putString("tool_mode", currentToolMode.name) }
    }

    val colorOptions = remember {
        arrayOf(
            Color.WHITE,
            Color.RED,
            Color.GREEN,
            Color.BLUE,
            Color.YELLOW,
            Color.CYAN,
            Color.MAGENTA,
            Color.BLACK
        )
    }

    val maskBitmap = remember {
        if (existingMaskBitmap != null) {
            existingMaskBitmap.copy(existingMaskBitmap.config ?: Bitmap.Config.ARGB_8888, true)
        } else {
            createBitmap(originalBitmap.width, originalBitmap.height).apply {
                val canvas = Canvas(this)
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
            }
        }
    }

    val tempBitmap = remember {
        createBitmap(originalBitmap.width, originalBitmap.height).apply {
            val canvas = Canvas(this)
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        }
    }
    var displayMaskBitmap by remember { mutableStateOf(tempBitmap.asImageBitmap()) }

    val androidPath = remember { Path() }
    var brushSizeDpValue by remember { mutableStateOf(30f) }
    var isDrawing by remember { mutableStateOf(false) }
    val currentPathPoints = remember { mutableStateListOf<Offset>() }

    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }

    val pathHistory = remember {
        mutableStateListOf<PathData>().apply {
            existingPathHistory?.let { addAll(it) }
        }
    }
    val redoStack = remember { mutableStateListOf<PathData>() }

    val brushPaint = remember {
        Paint().apply {
            color = brushColor
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
        }
    }

    val eraserPaint = remember {
        Paint().apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
            xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
        }
    }

    val finalPaint = remember {
        Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
        }
    }

    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var imageRect by remember { mutableStateOf<Rect?>(null) }

    var displayUpdateTrigger by remember { mutableStateOf(0) }

    fun updateAllBrushPaths(newColor: Int) {
        pathHistory.forEachIndexed { index, pathData ->
            if (pathData.mode == ToolMode.BRUSH) {
                pathHistory[index] = pathData.copy(color = newColor)
            }
        }
    }

    LaunchedEffect(brushColor) {
        brushPaint.color = brushColor
        updateAllBrushPaths(brushColor)
        sharedPrefs.edit { putInt("brush_color", brushColor) }
        displayUpdateTrigger++
    }

    fun mapToImageCoordinate(canvasPoint: Offset): Offset? {
        val rect = imageRect ?: return null
        if (!rect.contains(canvasPoint)) {
            val tolerance = 5f * density.density
            if (canvasPoint.x < rect.left - tolerance || canvasPoint.x > rect.right + tolerance ||
                canvasPoint.y < rect.top - tolerance || canvasPoint.y > rect.bottom + tolerance
            ) {
                return null
            }
        }
        val clampedX = canvasPoint.x.coerceIn(rect.left, rect.right)
        val clampedY = canvasPoint.y.coerceIn(rect.top, rect.bottom)
        val relativeX = (clampedX - rect.left) / rect.width
        val relativeY = (clampedY - rect.top) / rect.height
        return Offset(
            relativeX * originalBitmap.width,
            relativeY * originalBitmap.height
        )
    }

    fun convertDpToImagePixels(
        dpValue: Float,
        density: Density,
        imageRect: Rect?,
        originalWidth: Int
    ): Float {
        val rect = imageRect ?: return dpValue

        val brushSizeInScreenPx = with(density) { dpValue.dp.toPx() }

        val scale = if (rect.width > 0) originalWidth.toFloat() / rect.width else 1f

        return max(1f, brushSizeInScreenPx * scale)
    }

    fun updateDisplayMask() {
        val canvas = Canvas(tempBitmap)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        if (existingMaskBitmap != null) {
            val filterPaint = Paint().apply {
                colorFilter = android.graphics.PorterDuffColorFilter(brushColor, PorterDuff.Mode.SRC_IN)
            }
            canvas.drawBitmap(existingMaskBitmap, 0f, 0f, filterPaint)
        }

        pathHistory.forEach { pathData ->
            // SMART_SELECT/AUTO_DETECT masks are merged into maskBitmap, skip path drawing
            if (pathData.mode == ToolMode.SMART_SELECT || pathData.mode == ToolMode.AUTO_DETECT) return@forEach

            if (pathData.points.size > 1) {
                val paint = when (pathData.mode) {
                    ToolMode.BRUSH -> brushPaint.apply { color = pathData.color }
                    ToolMode.ERASER -> eraserPaint
                    ToolMode.SMART_SELECT -> return@forEach
                    ToolMode.AUTO_DETECT -> return@forEach
                }

                paint.strokeWidth = pathData.size

                androidPath.reset()
                androidPath.moveTo(pathData.points[0].x, pathData.points[0].y)
                for (i in 1 until pathData.points.size) {
                    androidPath.lineTo(pathData.points[i].x, pathData.points[i].y)
                }
                canvas.drawPath(androidPath, paint)
            }
        }

        // Draw the maskBitmap which contains SMART_SELECT masks
        val maskPaint = Paint().apply {
            colorFilter = android.graphics.PorterDuffColorFilter(brushColor, PorterDuff.Mode.SRC_IN)
        }
        canvas.drawBitmap(maskBitmap, 0f, 0f, maskPaint)

        if (isDrawing && currentPathPoints.size > 1) {
            val paint = when (currentToolMode) {
                ToolMode.BRUSH -> brushPaint
                ToolMode.ERASER -> eraserPaint
                ToolMode.SMART_SELECT -> null
                ToolMode.AUTO_DETECT -> null
            }
            if (paint == null) return // Non-drawing modes

            paint.strokeWidth = convertDpToImagePixels(
                brushSizeDpValue,
                density,
                imageRect,
                originalBitmap.width
            ) / scale
            androidPath.reset()
            androidPath.moveTo(currentPathPoints[0].x, currentPathPoints[0].y)
            for (i in 1 until currentPathPoints.size) {
                androidPath.lineTo(currentPathPoints[i].x, currentPathPoints[i].y)
            }
            canvas.drawPath(androidPath, paint)
        }

        displayMaskBitmap =
            tempBitmap.copy(tempBitmap.config ?: Bitmap.Config.ARGB_8888, true).asImageBitmap()
    }

    fun handleSmartSelectTap(imagePoint: Offset) {
        if (!isSamModelLoaded || isSamProcessing) return
        coroutineScope.launch {
            isSamProcessing = true
            samLoadingMessage = "Analyzing..."
            try {
                val mask = samSegmenter.segmentAtPoint(
                    originalBitmap,
                    imagePoint.x,
                    imagePoint.y
                )
                if (mask != null) {
                    // Merge SAM mask into the display mask (additive)
                    val canvas = Canvas(maskBitmap)
                    val paint = Paint().apply {
                        xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
                    }
                    canvas.drawBitmap(mask, 0f, 0f, paint)

                    // Add as a path entry for undo support
                    pathHistory.add(
                        PathData(
                            points = listOf(imagePoint),
                            size = 0f,
                            mode = ToolMode.SMART_SELECT,
                            color = Color.WHITE
                        )
                    )
                    redoStack.clear()
                    updateDisplayMask()
                    Log.i("InpaintScreen", "Smart Select mask merged for point (${imagePoint.x}, ${imagePoint.y})")
                }
            } catch (e: Exception) {
                Log.e("InpaintScreen", "Smart Select failed: ${e.message}", e)
            } finally {
                isSamProcessing = false
                samLoadingMessage = ""
            }
        }
    }

    fun scanObjects() {
        if (!isSamModelLoaded || isScanning) return
        coroutineScope.launch {
            isScanning = true
            scanProgress = "Scanning for objects..."
            try {
                val result = withContext(Dispatchers.Default) {
                    samSegmenter.segmentImage(originalBitmap, 32)
                }
                if (result != null && result.objects.isNotEmpty()) {
                    detectedObjects = result.objects
                    selectedObjectIds.clear()
                    Log.i("InpaintScreen", "Auto Detect found ${result.objects.size} objects")
                } else {
                    Log.w("InpaintScreen", "Auto Detect found no objects")
                }
            } catch (e: Exception) {
                Log.e("InpaintScreen", "Auto Detect failed: ${e.message}", e)
            } finally {
                isScanning = false
                scanProgress = ""
            }
        }
    }

    fun toggleObjectMask(obj: SegmentedObject) {
        if (selectedObjectIds.contains(obj.id)) {
            // Deselect: remove this object's mask
            selectedObjectIds.remove(obj.id)
        } else {
            // Select: add this object's mask
            selectedObjectIds.add(obj.id)
        }

        // Rebuild maskBitmap from all selected objects
        val canvas = Canvas(maskBitmap)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        for (selId in selectedObjectIds) {
            val selObj = detectedObjects.find { it.id == selId } ?: continue
            if (selObj.mask != null && !selObj.mask.isRecycled) {
                val srcW = min(selObj.mask.width, originalBitmap.width)
                val srcH = min(selObj.mask.height, originalBitmap.height)
                canvas.drawBitmap(
                    selObj.mask,
                    android.graphics.Rect(0, 0, srcW, srcH),
                    android.graphics.Rect(0, 0, originalBitmap.width, originalBitmap.height),
                    null
                )
            }
        }

        // Add a PathData entry for undo support
        pathHistory.add(
            PathData(
                points = emptyList(),
                size = 0f,
                mode = ToolMode.AUTO_DETECT,
                color = Color.WHITE
            )
        )
        redoStack.clear()
        updateDisplayMask()
    }

    // Small facial features that SAM can't isolate - use synthetic masks instead
    val syntheticMaskRegions = setOf(
        SmartPromptProcessor.TargetRegion.EYES,
        SmartPromptProcessor.TargetRegion.NOSE,
        SmartPromptProcessor.TargetRegion.MOUTH,
        SmartPromptProcessor.TargetRegion.EARS
    )

    // Synthetic mask proportions relative to face bbox
    // (centerX%, centerY%, width%, height%) - all relative to face bounding box
    val facialFeatureProportions = mapOf(
        SmartPromptProcessor.TargetRegion.EYES to FacialMaskRect(0.5f, 0.38f, 0.75f, 0.15f),
        SmartPromptProcessor.TargetRegion.NOSE to FacialMaskRect(0.5f, 0.58f, 0.25f, 0.20f),
        SmartPromptProcessor.TargetRegion.MOUTH to FacialMaskRect(0.5f, 0.75f, 0.45f, 0.15f),
        SmartPromptProcessor.TargetRegion.EARS to FacialMaskRect(0.5f, 0.42f, 0.95f, 0.18f)
    )

    fun processSmartEdit() {
        if (smartEditText.isBlank()) return
        val processed = SmartPromptProcessor.process(smartEditText)
        processedPrompt = processed
        Log.i("InpaintScreen", "Smart Edit: '${smartEditText}' → target=${processed.targetRegion.label}, prompt='${processed.inpaintPrompt}', denoise=${processed.denoiseStrength}")

        coroutineScope.launch {
            isLoading = true
            isSamProcessing = true
            samLoadingMessage = "Finding ${processed.targetRegion.label}..."
            try {
                val region = processed.targetRegion
                val w = originalBitmap.width.toFloat()
                val h = originalBitmap.height.toFloat()
                val isFacialFeature = region in syntheticMaskRegions

                // ── PASS 1: Find the main person/subject ──
                samLoadingMessage = "Finding subject..."
                Log.i("InpaintScreen", "Pass 1: Finding main subject at center")
                val personMask = withContext(Dispatchers.Default) {
                    samSegmenter.segmentAtPoint(originalBitmap, w / 2f, h / 2f)
                }

                if (personMask == null) {
                    Log.w("InpaintScreen", "Pass 1 failed")
                    samLoadingMessage = "Could not find subject"
                    kotlinx.coroutines.delay(2000)
                    return@launch
                }

                // Calculate person bounding box
                var pMinX = personMask.width; var pMinY = personMask.height
                var pMaxX = 0; var pMaxY = 0
                for (py in 0 until personMask.height) {
                    for (px in 0 until personMask.width) {
                        if (personMask.getPixel(px, py) != Color.TRANSPARENT) {
                            if (px < pMinX) pMinX = px; if (px > pMaxX) pMaxX = px
                            if (py < pMinY) pMinY = py; if (py > pMaxY) pMaxY = py
                        }
                    }
                }
                val personW = (pMaxX - pMinX).toFloat()
                val personH = (pMaxY - pMinY).toFloat()
                Log.i("InpaintScreen", "Pass 1: Person bbox ($pMinX,$pMinY)-($pMaxX,$pMaxY) ${personW}x${personH}")

                if (personW < 10 || personH < 10) {
                    samLoadingMessage = "Subject too small"
                    kotlinx.coroutines.delay(2000)
                    return@launch
                }

                val resultMask: Bitmap

                if (isFacialFeature) {
                    // ── PASS 2: Find the FACE within the person ──
                    samLoadingMessage = "Finding face..."
                    // Estimate face position: top 30% of person, centered
                    val facePointX = pMinX + personW * 0.5f
                    val facePointY = pMinY + personH * 0.18f  // upper portion

                    Log.i("InpaintScreen", "Pass 2: Finding face at ($facePointX, $facePointY)")

                    // Use SAM with negatives to isolate face from body
                    val facePositive = listOf(Pair(facePointX, facePointY))
                    val faceNegative = listOf(
                        Pair(pMinX + personW * 0.5f, pMinY + personH * 0.55f),  // chest
                        Pair(pMinX + personW * 0.5f, pMinY + personH * 0.75f),  // waist
                        Pair(pMinX + personW * 0.15f, pMinY + personH * 0.45f), // left arm
                        Pair(pMinX + personW * 0.85f, pMinY + personH * 0.45f)  // right arm
                    )

                    val faceMask = withContext(Dispatchers.Default) {
                        samSegmenter.segmentPrecise(originalBitmap, facePositive, faceNegative)
                    }

                    if (faceMask == null) {
                        Log.w("InpaintScreen", "Pass 2: Face detection failed")
                        samLoadingMessage = "Could not find face"
                        kotlinx.coroutines.delay(2000)
                        return@launch
                    }

                    // Get face bounding box
                    var fMinX = faceMask.width; var fMinY = faceMask.height
                    var fMaxX = 0; var fMaxY = 0
                    for (py in 0 until faceMask.height) {
                        for (px in 0 until faceMask.width) {
                            if (faceMask.getPixel(px, py) != Color.TRANSPARENT) {
                                if (px < fMinX) fMinX = px; if (px > fMaxX) fMaxX = px
                                if (py < fMinY) fMinY = py; if (py > fMaxY) fMaxY = py
                            }
                        }
                    }
                    val faceW = (fMaxX - fMinX).toFloat()
                    val faceH = (fMaxY - fMinY).toFloat()
                    Log.i("InpaintScreen", "Pass 2: Face bbox ($fMinX,$fMinY)-($fMaxX,$fMaxY) ${faceW}x${faceH}")

                    // ── PASS 3: Create synthetic elliptical mask for the specific feature ──
                    samLoadingMessage = "Masking ${region.label}..."
                    val props = facialFeatureProportions[region]
                    if (props == null || faceW < 5 || faceH < 5) {
                        samLoadingMessage = "Cannot target ${region.label}"
                        kotlinx.coroutines.delay(2000)
                        return@launch
                    }

                    // Calculate ellipse center and size in absolute pixels
                    val cx = fMinX + faceW * props.centerXPct
                    val cy = fMinY + faceH * props.centerYPct
                    val rx = faceW * props.widthPct / 2f  // horizontal radius
                    val ry = faceH * props.heightPct / 2f // vertical radius

                    Log.i("InpaintScreen", "Pass 3: Synthetic ${region.label} ellipse center=($cx,$cy) radii=($rx,$ry)")

                    // Draw the synthetic mask
                    resultMask = Bitmap.createBitmap(originalBitmap.width, originalBitmap.height, Bitmap.Config.ARGB_8888)
                    val maskCanvas = Canvas(resultMask)
                    val paint = Paint().apply {
                        color = Color.WHITE
                        style = Paint.Style.FILL
                        isAntiAlias = true
                    }
                    val rect = android.graphics.RectF(cx - rx, cy - ry, cx + rx, cy + ry)
                    maskCanvas.drawOval(rect, paint)

                    // Count white pixels
                    var maskPx = 0
                    for (py in 0 until resultMask.height) {
                        for (px in 0 until resultMask.width) {
                            if (resultMask.getPixel(px, py) != Color.TRANSPARENT) maskPx++
                        }
                    }
                    Log.i("InpaintScreen", "Pass 3: Synthetic mask = ${maskPx}px (${String.format("%.1f", maskPx * 100f / (w * h))}%)")

                } else {
                    // ── Large region: Use SAM directly (face, hair, shirt, etc.) ──
                    samLoadingMessage = "Targeting ${region.label}..."
                    val targetX = pMinX + personW * region.relativeX
                    val targetY = pMinY + personH * region.relativeY

                    val negativePoints = region.negativeOffsets.map { (nx, ny) ->
                        Pair(pMinX + personW * nx, pMinY + personH * ny)
                    }
                    val positivePoints = listOf(Pair(targetX, targetY))

                    Log.i("InpaintScreen", "SAM targeting '${region.label}' at ($targetX, $targetY)")

                    val samMask = if (negativePoints.isNotEmpty()) {
                        withContext(Dispatchers.Default) {
                            samSegmenter.segmentPrecise(originalBitmap, positivePoints, negativePoints)
                        }
                    } else {
                        withContext(Dispatchers.Default) {
                            samSegmenter.segmentAtPoint(originalBitmap, targetX, targetY)
                        }
                    }

                    if (samMask == null) {
                        samLoadingMessage = "Could not find ${region.label}"
                        kotlinx.coroutines.delay(2000)
                        return@launch
                    }
                    resultMask = samMask
                }

                // Apply final mask
                val finalMaskCanvas = Canvas(maskBitmap)
                finalMaskCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                finalMaskCanvas.drawBitmap(resultMask, 0f, 0f, null)
                updateDisplayMask()

                // Encode mask to base64
                val base64String = withContext(Dispatchers.Default) {
                    val byteArrayOutputStream = ByteArrayOutputStream()
                    maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                    val byteArray = byteArrayOutputStream.toByteArray()
                    Base64.getEncoder().encodeToString(byteArray)
                }

                if (onSmartEdit != null) {
                    onSmartEdit.invoke(
                        base64String,
                        processed.inpaintPrompt,
                        processed.negativePrompt,
                        processed.denoiseStrength,
                        processed.cfgScale
                    )
                } else {
                    onInpaintComplete(base64String, originalBitmap, maskBitmap, pathHistory.toList())
                }
            } catch (e: Exception) {
                Log.e("InpaintScreen", "Smart Edit error: ${e.message}", e)
            } finally {
                isLoading = false
                isSamProcessing = false
                samLoadingMessage = ""
            }
        }
    }

    fun processMask() {
        coroutineScope.launch {
            isLoading = true
            errorMessage = null
            try {
                val finalMaskCanvas = Canvas(maskBitmap)
                finalMaskCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                if (existingMaskBitmap != null) {
                    finalMaskCanvas.drawBitmap(existingMaskBitmap, 0f, 0f, null)
                }

                for (pathData in pathHistory) {
                    // AUTO_DETECT entries are handled by maskBitmap rebuild, skip
                    if (pathData.mode == ToolMode.AUTO_DETECT) continue
                    if (pathData.points.isEmpty()) continue

                    // SMART_SELECT: re-run SAM to rebuild the mask
                    if (pathData.mode == ToolMode.SMART_SELECT) {
                        val samMask = kotlinx.coroutines.runBlocking {
                            samSegmenter.segmentAtPoint(
                                originalBitmap,
                                pathData.points[0].x,
                                pathData.points[0].y
                            )
                        }
                        if (samMask != null) {
                            finalMaskCanvas.drawBitmap(samMask, 0f, 0f, null)
                        }
                        continue
                    }

                    val paint = when (pathData.mode) {
                        ToolMode.BRUSH -> finalPaint
                        ToolMode.ERASER -> eraserPaint
                        ToolMode.SMART_SELECT -> continue
                        ToolMode.AUTO_DETECT -> continue
                    }

                    paint.strokeWidth = pathData.size

                    androidPath.reset()
                    androidPath.moveTo(pathData.points[0].x, pathData.points[0].y)
                    for (i in 1 until pathData.points.size) {
                        androidPath.lineTo(pathData.points[i].x, pathData.points[i].y)
                    }
                    finalMaskCanvas.drawPath(androidPath, paint)
                }

                val base64String = withContext(Dispatchers.IO) {
                    val byteArrayOutputStream = ByteArrayOutputStream()
                    maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                    val byteArray = byteArrayOutputStream.toByteArray()
                    Base64.getEncoder().encodeToString(byteArray)
                }
                onInpaintComplete(base64String, originalBitmap, maskBitmap, pathHistory.toList())
            } catch (e: Exception) {
                errorMessage = "Error: ${e.message}"
                e.printStackTrace()
            } finally {
                isLoading = false
            }
        }
    }

    fun undoLastPath() {
        if (pathHistory.isNotEmpty()) {
            val lastPath = pathHistory.removeAt(pathHistory.size - 1)
            redoStack.add(lastPath)
            updateDisplayMask()
        }
    }

    fun redoLastPath() {
        if (redoStack.isNotEmpty()) {
            val pathToRedo = redoStack.removeAt(redoStack.size - 1)
            pathHistory.add(pathToRedo)
            updateDisplayMask()
        }
    }

    BackHandler { onCancel() }

    LaunchedEffect(
        pathHistory.size,
        currentPathPoints.size,
        isDrawing,
        imageRect,
        density,
        displayUpdateTrigger,
        currentToolMode,
        scale
    ) {
        updateDisplayMask()
    }

    if (showColorPicker) {
        AlertDialog(
            onDismissRequest = { showColorPicker = false },
            title = { Text(stringResource(R.string.brush_color)) },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    LazyVerticalGrid(
                        columns = GridCells.Fixed(4),
                        contentPadding = PaddingValues(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        items(colorOptions.size) { index ->
                            val color = colorOptions[index]
                            Box(
                                modifier = Modifier
                                    .padding(8.dp)
                                    .size(40.dp)
                                    .clip(CircleShape)
                                    .background(ComposeColor(color))
                                    .border(
                                        width = 2.dp,
                                        color = if (color == brushColor) ComposeColor.Black else ComposeColor.Transparent,
                                        shape = CircleShape
                                    )
                                    .clickable {
                                        brushColor = color
                                        showColorPicker = false
                                    }
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showColorPicker = false }) {
                    Text(stringResource(R.string.close))
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.set_inpaint_area)) },
                navigationIcon = {
                    IconButton(onClick = onCancel) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            if (!isLoading) processMask()
                        },
                        enabled = !isLoading
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "Complete Marking"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(16.dp)
                        .clipToBounds()
                        .onGloballyPositioned { coordinates ->
                            val size = coordinates.size
                            val imageWidth = size.width.toFloat()
                            val imageHeight = size.height.toFloat()
                            val originalAspect =
                                originalBitmap.width.toFloat() / originalBitmap.height.toFloat()
                            val boxAspect = imageWidth / imageHeight
                            val scaledWidth: Float
                            val scaledHeight: Float
                            if (originalAspect > boxAspect) {
                                scaledWidth = imageWidth
                                scaledHeight = imageWidth / originalAspect
                            } else {
                                scaledHeight = imageHeight
                                scaledWidth = imageHeight * originalAspect
                            }
                            val left = (imageWidth - scaledWidth) / 2
                            val top = (imageHeight - scaledHeight) / 2
                            val newRect =
                                Rect(left, top, left + scaledWidth, bottom = top + scaledHeight)
                            if (newRect != imageRect) {
                                imageRect = newRect
                            }
                        }
                        .pointerInput(Unit) {
                            val containerWidth = size.width.toFloat()
                            val containerHeight = size.height.toFloat()

                            awaitEachGesture {
                                val firstDown = awaitFirstDown(requireUnconsumed = false)
                                var isMultiTouch = false
                                var drawStarted = false

                                fun touchToContent(pos: Offset): Offset {
                                    val pivotX = containerWidth / 2f
                                    val pivotY = containerHeight / 2f
                                    return Offset(
                                        (pos.x - offsetX - pivotX) / scale + pivotX,
                                        (pos.y - offsetY - pivotY) / scale + pivotY
                                    )
                                }

                                val contentPos = touchToContent(firstDown.position)
                                val firstImgPt = mapToImageCoordinate(contentPos)

                                // Smart Select: single tap to segment
                                if (currentToolMode == ToolMode.SMART_SELECT && firstImgPt != null) {
                                    handleSmartSelectTap(firstImgPt)
                                    return@awaitEachGesture
                                }

                                // Auto Detect: no drawing, just consume gesture
                                if (currentToolMode == ToolMode.AUTO_DETECT) {
                                    return@awaitEachGesture
                                }

                                if (firstImgPt != null) {
                                    drawStarted = true
                                    isDrawing = true
                                    currentPathPoints.clear()
                                    currentPathPoints.add(firstImgPt)
                                }

                                var prevPointers: List<Offset> = listOf(firstDown.position)

                                while (true) {
                                    val event = awaitPointerEvent()
                                    val activeChanges = event.changes.filter { it.pressed }

                                    if (activeChanges.isEmpty()) {
                                        if (drawStarted && !isMultiTouch) {
                                            isDrawing = false
                                            if (currentPathPoints.isNotEmpty()) {
                                                val imgPxSize = convertDpToImagePixels(
                                                    brushSizeDpValue,
                                                    density,
                                                    imageRect,
                                                    originalBitmap.width
                                                ) / scale
                                                pathHistory.add(
                                                    PathData(
                                                        points = currentPathPoints.toList(),
                                                        size = imgPxSize,
                                                        mode = currentToolMode,
                                                        color = brushColor
                                                    )
                                                )
                                                currentPathPoints.clear()
                                                redoStack.clear()
                                            }
                                        } else {
                                            isDrawing = false
                                            currentPathPoints.clear()
                                        }
                                        break
                                    }

                                    if (activeChanges.size >= 2) {
                                        if (!isMultiTouch) {
                                            isMultiTouch = true
                                            isDrawing = false
                                            currentPathPoints.clear()
                                            drawStarted = false
                                            prevPointers = activeChanges.map { it.position }
                                        }

                                        val positions = activeChanges.map { it.position }
                                        val prevCentroid = Offset(
                                            prevPointers.map { it.x }.average().toFloat(),
                                            prevPointers.map { it.y }.average().toFloat()
                                        )
                                        val currCentroid = Offset(
                                            positions.map { it.x }.average().toFloat(),
                                            positions.map { it.y }.average().toFloat()
                                        )

                                        val prevSpread = if (prevPointers.size >= 2) {
                                            val dx = prevPointers[0].x - prevPointers[1].x
                                            val dy = prevPointers[0].y - prevPointers[1].y
                                            sqrt(dx * dx + dy * dy)
                                        } else 0f
                                        val currSpread = if (positions.size >= 2) {
                                            val dx = positions[0].x - positions[1].x
                                            val dy = positions[0].y - positions[1].y
                                            sqrt(dx * dx + dy * dy)
                                        } else 0f

                                        val zoomFactor =
                                            if (prevSpread > 10f) currSpread / prevSpread else 1f
                                        val panDelta = currCentroid - prevCentroid

                                        val newScale = (scale * zoomFactor).coerceIn(1f, 5f)
                                        val effectiveZoom = newScale / scale

                                        val pivotX = containerWidth / 2f
                                        val pivotY = containerHeight / 2f
                                        val newOffsetX =
                                            (currCentroid.x - pivotX) * (1f - effectiveZoom) + offsetX * effectiveZoom + panDelta.x
                                        val newOffsetY =
                                            (currCentroid.y - pivotY) * (1f - effectiveZoom) + offsetY * effectiveZoom + panDelta.y

                                        scale = newScale
                                        offsetX = newOffsetX
                                        offsetY = newOffsetY

                                        prevPointers = positions
                                        activeChanges.forEach { it.consume() }
                                    } else if (!isMultiTouch && drawStarted) {
                                        val change = activeChanges.first()
                                        val cPos = touchToContent(change.position)
                                        val imgPt = mapToImageCoordinate(cPos)
                                        if (imgPt != null) {
                                            currentPathPoints.add(imgPt)
                                        }
                                        change.consume()
                                        prevPointers = listOf(change.position)
                                    }
                                }
                            }
                        }
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .graphicsLayer(
                                scaleX = scale,
                                scaleY = scale,
                                translationX = offsetX,
                                translationY = offsetY
                            )
                    ) {
                        Image(
                            bitmap = originalBitmap.asImageBitmap(),
                            contentDescription = "Original Image",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize()
                        )

                        Canvas(
                            modifier = Modifier.fillMaxSize()
                        ) {
                            val rect = imageRect ?: return@Canvas
                            drawImage(
                                image = displayMaskBitmap,
                                srcOffset = IntOffset.Zero,
                                srcSize = IntSize(tempBitmap.width, tempBitmap.height),
                                dstOffset = IntOffset(rect.left.toInt(), rect.top.toInt()),
                                dstSize = IntSize(rect.width.toInt(), rect.height.toInt()),
                                alpha = 0.6f
                            )
                        }
                    }
                }

                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    shape = MaterialTheme.shapes.medium,
                    tonalElevation = 4.dp
                ) {
                    Column(
                        modifier = Modifier
                            .padding(16.dp)
                            .fillMaxWidth()
                    ) {
                        Text(
                            stringResource(R.string.inpaint_hint),
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Row(
                                modifier = Modifier.width(120.dp),
                                horizontalArrangement = Arrangement.SpaceEvenly
                            ) {
                                Box(
                                    modifier = Modifier.size(36.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (currentToolMode == ToolMode.BRUSH) {
                                        Box(
                                            modifier = Modifier
                                                .size(36.dp)
                                                .clip(CircleShape)
                                                .background(
                                                    MaterialTheme.colorScheme.primary.copy(
                                                        alpha = 0.15f
                                                    )
                                                )
                                        )
                                    }

                                    IconButton(
                                        onClick = { currentToolMode = ToolMode.BRUSH },
                                        modifier = Modifier.size(36.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.Brush,
                                            contentDescription = "Brush Tool",
                                            tint = if (currentToolMode == ToolMode.BRUSH)
                                                MaterialTheme.colorScheme.primary
                                            else
                                                MaterialTheme.colorScheme.onSurface,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                }

                                Box(
                                    modifier = Modifier.size(36.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (currentToolMode == ToolMode.ERASER) {
                                        Box(
                                            modifier = Modifier
                                                .size(36.dp)
                                                .clip(CircleShape)
                                                .background(
                                                    MaterialTheme.colorScheme.primary.copy(
                                                        alpha = 0.15f
                                                    )
                                                )
                                        )
                                    }

                                    IconButton(
                                        onClick = { currentToolMode = ToolMode.ERASER },
                                        modifier = Modifier.size(36.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.FormatPaint,
                                            contentDescription = "Eraser Tool",
                                            tint = if (currentToolMode == ToolMode.ERASER)
                                                MaterialTheme.colorScheme.primary
                                            else
                                                MaterialTheme.colorScheme.onSurface,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                }

                                // Smart Select (MobileSAM tap-to-segment)
                                Box(
                                    modifier = Modifier.size(36.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (currentToolMode == ToolMode.SMART_SELECT) {
                                        Box(
                                            modifier = Modifier
                                                .size(36.dp)
                                                .clip(CircleShape)
                                                .background(
                                                    MaterialTheme.colorScheme.primary.copy(
                                                        alpha = 0.15f
                                                    )
                                                )
                                        )
                                    }

                                    IconButton(
                                        onClick = { currentToolMode = ToolMode.SMART_SELECT },
                                        modifier = Modifier.size(36.dp),
                                        enabled = isSamModelLoaded
                                    ) {
                                        Icon(
                                            Icons.Default.AutoFixHigh,
                                            contentDescription = "Smart Select",
                                            tint = if (currentToolMode == ToolMode.SMART_SELECT)
                                                MaterialTheme.colorScheme.primary
                                            else if (!isSamModelLoaded)
                                                MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                                            else
                                                MaterialTheme.colorScheme.onSurface,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                }

                                // Smart Edit (prompt-driven auto-mask)
                                Box(
                                    modifier = Modifier.size(36.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (currentToolMode == ToolMode.AUTO_DETECT) {
                                        Box(
                                            modifier = Modifier
                                                .size(36.dp)
                                                .clip(CircleShape)
                                                .background(
                                                    MaterialTheme.colorScheme.primary.copy(
                                                        alpha = 0.15f
                                                    )
                                                )
                                        )
                                    }

                                    IconButton(
                                        onClick = { currentToolMode = ToolMode.AUTO_DETECT },
                                        modifier = Modifier.size(36.dp),
                                        enabled = isSamModelLoaded
                                    ) {
                                        Icon(
                                            Icons.Default.GridView,
                                            contentDescription = "Auto Detect Objects",
                                            tint = if (currentToolMode == ToolMode.AUTO_DETECT)
                                                MaterialTheme.colorScheme.primary
                                            else if (!isSamModelLoaded)
                                                MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                                            else
                                                MaterialTheme.colorScheme.onSurface,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                }
                            }

                            // Only show slider and color when in Brush/Eraser mode
                            if (currentToolMode == ToolMode.BRUSH || currentToolMode == ToolMode.ERASER) {
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .padding(horizontal = 8.dp)
                            ) {
                                Slider(
                                    value = brushSizeDpValue,
                                    onValueChange = { brushSizeDpValue = it },
                                    valueRange = 5f..50f,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }

                            Box(
                                modifier = Modifier
                                    .width(50.dp)
                                    .padding(start = 4.dp)
                                    .aspectRatio(1f),
                                contentAlignment = Alignment.Center
                            ) {
                                val indicatorSize = with(density) { brushSizeDpValue.dp }
                                Box(
                                    modifier = Modifier
                                        .size(indicatorSize.coerceAtMost(50.dp))
                                        .clip(CircleShape)
                                        .background(
                                            if (currentToolMode == ToolMode.BRUSH)
                                                ComposeColor(brushColor)
                                            else
                                                ComposeColor.LightGray.copy(alpha = 0.5f)
                                        )
                                        .border(
                                            width = 1.dp,
                                            color = ComposeColor.DarkGray.copy(alpha = 0.3f),
                                            shape = CircleShape
                                        )
                                        .clickable(enabled = currentToolMode == ToolMode.BRUSH) {
                                            if (currentToolMode == ToolMode.BRUSH) {
                                                showColorPicker = true
                                            }
                                        }
                                )
                            }
                            } else if (currentToolMode == ToolMode.SMART_SELECT) {
                                // Smart Select mode: show hint text
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .padding(horizontal = 8.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        if (isSamProcessing) "Analyzing..."
                                        else "Tap on object to select",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                                        fontSize = 12.sp
                                    )
                                }
                            } else if (currentToolMode == ToolMode.AUTO_DETECT) {
                                // Smart Edit: prompt-first auto-detect
                                Column(
                                    modifier = Modifier
                                        .weight(1f)
                                        .padding(horizontal = 4.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        OutlinedTextField(
                                            value = smartEditText,
                                            onValueChange = {
                                                smartEditText = it
                                                if (it.length > 3) {
                                                    processedPrompt = SmartPromptProcessor.process(it)
                                                } else {
                                                    processedPrompt = null
                                                }
                                            },
                                            modifier = Modifier.weight(1f),
                                            placeholder = {
                                                Text(
                                                    "e.g. change her eyes to red",
                                                    style = MaterialTheme.typography.bodySmall,
                                                    fontSize = 11.sp
                                                )
                                            },
                                            textStyle = MaterialTheme.typography.bodySmall.copy(fontSize = 13.sp),
                                            singleLine = true,
                                            shape = RoundedCornerShape(12.dp),
                                            colors = OutlinedTextFieldDefaults.colors(
                                                focusedBorderColor = MaterialTheme.colorScheme.primary,
                                                unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.4f)
                                            )
                                        )
                                        Button(
                                            onClick = { processSmartEdit() },
                                            enabled = smartEditText.isNotBlank() && !isLoading && !isSamProcessing,
                                            shape = RoundedCornerShape(12.dp),
                                            contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp)
                                        ) {
                                            Icon(
                                                Icons.Default.AutoFixHigh,
                                                contentDescription = "Apply Smart Edit",
                                                modifier = Modifier.size(18.dp)
                                            )
                                            Spacer(modifier = Modifier.width(4.dp))
                                            Text("Go", fontSize = 13.sp)
                                        }
                                    }
                                    if (processedPrompt != null) {
                                        Text(
                                            "Target: ${processedPrompt!!.targetRegion.label} → ${processedPrompt!!.inpaintPrompt.take(50)}...",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                                            fontSize = 10.sp,
                                            maxLines = 1,
                                            modifier = Modifier.padding(top = 2.dp, start = 4.dp)
                                        )
                                    }
                                }
                            }
                        }

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 12.dp),
                            horizontalArrangement = Arrangement.spacedBy(
                                16.dp,
                                Alignment.CenterHorizontally
                            )
                        ) {
                            Button(
                                onClick = { undoLastPath() },
                                enabled = pathHistory.isNotEmpty() && !isLoading
                            ) {
                                Icon(
                                    Icons.Default.Refresh,
                                    contentDescription = "Undo",
                                    modifier = Modifier.size(ButtonDefaults.IconSize)
                                )
                                Spacer(modifier = Modifier.width(ButtonDefaults.IconSpacing))
                                Text(stringResource(R.string.undo))
                            }

                            Button(
                                onClick = { redoLastPath() },
                                enabled = redoStack.isNotEmpty() && !isLoading,
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                            ) {
                                Icon(
                                    Icons.Default.Redo,
                                    contentDescription = "Redo",
                                    modifier = Modifier.size(ButtonDefaults.IconSize)
                                )
                                Spacer(modifier = Modifier.width(ButtonDefaults.IconSpacing))
                                Text(stringResource(R.string.redo))
                            }
                        }
                    }
                }
            }

            if (isLoading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(ComposeColor.Black.copy(alpha = 0.6f))
                        .pointerInput(Unit) {},
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(64.dp),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                }
            }

            if (isSamProcessing) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(ComposeColor.Black.copy(alpha = 0.3f))
                        .pointerInput(Unit) {},
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        shape = MaterialTheme.shapes.medium,
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.95f)
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(24.dp),
                                strokeWidth = 2.dp
                            )
                            Text(
                                "Analyzing...",
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            }

            if (errorMessage != null) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = MaterialTheme.shapes.large,
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                    ) {
                        Column(modifier = Modifier.padding(24.dp)) {
                            Text(
                                text = "Error",
                                style = MaterialTheme.typography.headlineSmall,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = errorMessage ?: "Unknown error occurred",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = { errorMessage = null },
                                modifier = Modifier.align(Alignment.End),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.error,
                                    contentColor = MaterialTheme.colorScheme.onError
                                )
                            ) {
                                Text("OK")
                            }
                        }
                    }
                }
            }
        }
    }
}