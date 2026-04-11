package io.github.xororz.localdream.ui.screens

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.xororz.localdream.R
import io.github.xororz.localdream.ml.BoundingBox
import io.github.xororz.localdream.ml.ImageObjects
import io.github.xororz.localdream.ml.SegmentedObject
import io.github.xororz.localdream.ml.MobileSAMSegmenter
import io.github.xororz.localdream.ml.SimpleSegmenter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.util.Log
import kotlin.math.min

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ObjectSelectionScreen(
    originalBitmap: Bitmap,
    onObjectSelected: (Bitmap, SegmentedObject) -> Unit,
    onAnalyzeComplete: (ImageObjects) -> Unit,
    onCancel: () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val context = androidx.compose.ui.platform.LocalContext.current
    val density = LocalDensity.current
    
    var isAnalyzing by remember { mutableStateOf(false) }
    var imageObjects by remember { mutableStateOf<ImageObjects?>(null) }
    var selectedObjectId by remember { mutableStateOf<Int?>(null) }
    var showJson by remember { mutableStateOf(false) }
    
    var displayBitmap by remember { mutableStateOf(originalBitmap.asImageBitmap()) }
    var imageRect by remember { mutableStateOf<androidx.compose.ui.geometry.Rect?>(null) }
    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }
    
    val segmenter = remember { SimpleSegmenter(context) }
    val mobileSAMSegmenter = remember { MobileSAMSegmenter(context) }
    var isModelLoaded by remember { mutableStateOf(false) }
    
    LaunchedEffect(Unit) {
        isModelLoaded = mobileSAMSegmenter.loadModel()
    }
    
    fun analyzeImage() {
        coroutineScope.launch {
            isAnalyzing = true
            try {
                val result = withContext(Dispatchers.Default) {
                    segmenter.segmentImage(originalBitmap)
                }
                imageObjects = result
                onAnalyzeComplete(result)
            } catch (e: Exception) {
                Log.e("ObjectSelection", "Analysis failed: ${e.message}")
                e.printStackTrace()
            } finally {
                isAnalyzing = false
            }
        }
    }
    
    fun analyzeImageWithSAM() {
        coroutineScope.launch {
            isAnalyzing = true
            try {
                Log.i("ObjectSelection", "Starting SAM analysis...")
                Log.i("ObjectSelection", "Bitmap: ${originalBitmap.width}x${originalBitmap.height}")
                val result = withContext(Dispatchers.Default) {
                    mobileSAMSegmenter.segmentImage(originalBitmap, 32)
                }
                Log.i("ObjectSelection", "SAM analysis completed, result: ${result != null}")
                if (result != null) {
                    imageObjects = result
                    onAnalyzeComplete(result)
                    Log.i("ObjectSelection", "SAM found ${result.objects.size} objects")
                } else {
                    Log.w("ObjectSelection", "SAM returned null")
                }
            } catch (e: Exception) {
                Log.e("ObjectSelection", "SAM analysis failed: ${e.message}")
                e.printStackTrace()
            } finally {
                isAnalyzing = false
            }
        }
    }
    
    fun selectObject(obj: SegmentedObject) {
        selectedObjectId = obj.id
        
        val maskBitmap = Bitmap.createBitmap(
            originalBitmap.width,
            originalBitmap.height,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(maskBitmap)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        
        if (obj.maskData.isNotEmpty() && obj.maskData.first() == -1) {
            if (obj.mask != null && !obj.mask.isRecycled) {
                val srcX = 0
                val srcY = 0
                val srcW = min(obj.mask.width, originalBitmap.width)
                val srcH = min(obj.mask.height, originalBitmap.height)
                canvas.drawBitmap(obj.mask, android.graphics.Rect(srcX, srcY, srcW, srcH), 
                    android.graphics.Rect(0, 0, originalBitmap.width, originalBitmap.height), null)
            }
        } else {
            for (y in 0 until originalBitmap.height) {
                for (x in 0 until originalBitmap.width) {
                    if (obj.maskData.contains(y * originalBitmap.width + x)) {
                        maskBitmap.setPixel(x, y, Color.WHITE)
                    }
                }
            }
        }
        
        onObjectSelected(maskBitmap, obj)
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.select_object)) },
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
                        onClick = { analyzeImageWithSAM() },
                        enabled = !isAnalyzing
                    ) {
                        Icon(
                            imageVector = Icons.Default.AutoAwesome,
                            contentDescription = "SAM Analyze"
                        )
                    }
                    IconButton(
                        onClick = { showJson = !showJson }
                    ) {
                        Icon(
                            imageVector = Icons.Default.Code,
                            contentDescription = "Show JSON"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            Box(
                modifier = Modifier
                    .weight(0.6f)
                    .fillMaxHeight()
                    .padding(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(RoundedCornerShape(8.dp))
                        .border(1.dp, ComposeColor.Gray, RoundedCornerShape(8.dp))
                        .onGloballyPositioned { coords ->
                            val size = coords.size
                            val imgWidth = size.width.toFloat()
                            val imgHeight = size.height.toFloat()
                            val originalAspect = originalBitmap.width.toFloat() / originalBitmap.height.toFloat()
                            val boxAspect = imgWidth / imgHeight
                            
                            val scaledWidth: Float
                            val scaledHeight: Float
                            
                            if (originalAspect > boxAspect) {
                                scaledWidth = imgWidth
                                scaledHeight = imgWidth / originalAspect
                            } else {
                                scaledHeight = imgHeight
                                scaledWidth = imgHeight * originalAspect
                            }
                            
                            val left = (imgWidth - scaledWidth) / 2
                            val top = (imgHeight - scaledHeight) / 2
                            imageRect = androidx.compose.ui.geometry.Rect(left, top, left + scaledWidth, top + scaledHeight)
                        }
                        .pointerInput(Unit) {
                            detectTapGestures { offset ->
                                imageObjects?.objects?.forEach { obj ->
                                    val bbox = obj.bbox
                                    val rect = imageRect ?: return@detectTapGestures
                                    
                                    val imgX = ((offset.x - rect.left) / rect.width * originalBitmap.width).toInt()
                                    val imgY = ((offset.y - rect.top) / rect.height * originalBitmap.height).toInt()
                                    
                                    if (imgX in bbox.x..(bbox.x + bbox.width) &&
                                        imgY in bbox.y..(bbox.y + bbox.height)
                                    ) {
                                        selectObject(obj)
                                    }
                                }
                            }
                        }
                ) {
                    Image(
                        bitmap = displayBitmap,
                        contentDescription = "Image",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxSize()
                            .graphicsLayer(
                                scaleX = scale,
                                scaleY = scale,
                                translationX = offsetX,
                                translationY = offsetY
                            )
                    )
                    
                    imageObjects?.objects?.forEach { obj ->
                        val rect = imageRect ?: return@forEach
                        
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            val left = rect.left + (obj.bbox.x.toFloat() / originalBitmap.width * rect.width)
                            val top = rect.top + (obj.bbox.y.toFloat() / originalBitmap.height * rect.height)
                            val width = obj.bbox.width.toFloat() / originalBitmap.width * rect.width
                            val height = obj.bbox.height.toFloat() / originalBitmap.height * rect.height
                            
                            val isSelected = obj.id == selectedObjectId
                            
                            drawRect(
                                color = if (isSelected) ComposeColor(0xFF4CAF50) else ComposeColor(0x802196F3),
                                topLeft = Offset(left, top),
                                size = Size(width, height),
                                style = androidx.compose.ui.graphics.drawscope.Stroke(width = if (isSelected) 4f else 2f)
                            )
                            
                            if (isSelected) {
                                drawRect(
                                    color = ComposeColor(0x204CAF50),
                                    topLeft = Offset(left, top),
                                    size = Size(width, height)
                                )
                            }
                        }
                    }
                }
                
                if (isAnalyzing) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(ComposeColor.Black.copy(alpha = 0.5f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = ComposeColor.White)
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                "Analyzing image...",
                                color = ComposeColor.White,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }
            
            Column(
                modifier = Modifier
                    .weight(0.4f)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState())
                    .padding(8.dp)
            ) {
                if (showJson && imageObjects != null) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(
                                "JSON Output",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                imageObjects?.json ?: "",
                                style = MaterialTheme.typography.bodySmall,
                                fontSize = 10.sp
                            )
                        }
                    }
                } else {
                    Text(
                        "Detected Objects",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    imageObjects?.objects?.forEach { obj ->
                        ObjectCard(
                            obj = obj,
                            isSelected = obj.id == selectedObjectId,
                            onClick = { selectObject(obj) }
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                    
                    if (imageObjects == null && !isAnalyzing) {
                        Text(
                            "Tap Analyze to detect objects",
                            style = MaterialTheme.typography.bodyMedium,
                            color = ComposeColor.Gray
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ObjectCard(
    obj: SegmentedObject,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected)
                MaterialTheme.colorScheme.primaryContainer
            else
                MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    obj.label,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    "${(obj.confidence * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Row {
                Text(
                    "BBox: [${obj.bbox.x}, ${obj.bbox.y}, ${obj.bbox.width}x${obj.bbox.height}]",
                    style = MaterialTheme.typography.bodySmall,
                    fontSize = 10.sp
                )
            }
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Text(
                "Area: ${obj.maskData.size} pixels",
                style = MaterialTheme.typography.bodySmall,
                fontSize = 10.sp
            )
        }
    }
}
