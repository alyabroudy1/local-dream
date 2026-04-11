package io.github.xororz.localdream.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.abs

class SimpleSegmenter(private val context: Context) {
    
    companion object {
        private const val TAG = "SimpleSegmenter"
    }
    
    data class SegmentationResult(
        val mask: Bitmap,
        val label: String,
        val confidence: Float
    )
    
    suspend fun segmentImage(bitmap: Bitmap): ImageObjects = withContext(Dispatchers.Default) {
        val objects = mutableListOf<SegmentedObject>()
        
        val edgeMask = detectEdges(bitmap)
        val regions = findConnectedRegions(edgeMask)
        
        var objectId = 0
        for ((region, area) in regions) {
            if (area > 1000) {
                val bmpMask = createMaskFromRegion(bitmap.width, bitmap.height, region)
                val bbox = calculateBBox(region, bitmap.width, bitmap.height)
                
                val label = classifyRegion(area, bbox, bmpMask, bitmap)
                
                objects.add(
                    SegmentedObject(
                        id = objectId++,
                        label = label,
                        confidence = min(0.95f, area / 10000f),
                        bbox = bbox,
                        mask = bmpMask,
                        maskData = extractMaskIndices(region, bitmap.width)
                    )
                )
            }
        }
        
        val json = generateJsonGraph(objects, bitmap.width, bitmap.height)
        
        Log.i(TAG, "Found ${objects.size} objects in image")
        
        ImageObjects(
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
            objects = objects,
            json = json
        )
    }
    
    private fun detectEdges(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        val grayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscale)
        
        val paint = Paint()
        paint.colorFilter = android.graphics.ColorMatrixColorFilter(
            floatArrayOf(
                0.33f, 0.33f, 0.33f, 0f, 0f,
                0.33f, 0.33f, 0.33f, 0f, 0f,
                0.33f, 0.33f, 0.33f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f
            )
        )
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        
        val edges = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val edgeCanvas = Canvas(edges)
        
        for (x in 1 until width - 1) {
            for (y in 1 until height - 1) {
                val current = grayscale.getPixel(x, y)
                val right = grayscale.getPixel(x + 1, y)
                val bottom = grayscale.getPixel(x, y + 1)
                
                val diffH = abs(Color.red(current) - Color.red(right))
                val diffV = abs(Color.red(current) - Color.red(bottom))
                
                if (diffH > 30 || diffV > 30) {
                    edges.setPixel(x, y, Color.WHITE)
                }
            }
        }
        
        return edges
    }
    
    private fun findConnectedRegions(bitmap: Bitmap): List<Pair<Set<Pair<Int, Int>>, Int>> {
        val width = bitmap.width
        val height = bitmap.height
        val visited = Array(height) { BooleanArray(width) }
        val regions = mutableListOf<Pair<Set<Pair<Int, Int>>, Int>>()
        
        val stack = ArrayDeque<Pair<Int, Int>>()
        
        for (x in 0 until width) {
            for (y in 0 until height) {
                if (!visited[y][x] && bitmap.getPixel(x, y) == Color.WHITE) {
                    val region = mutableSetOf<Pair<Int, Int>>()
                    
                    stack.clear()
                    stack.add(Pair(x, y))
                    visited[y][x] = true
                    
                    while (stack.isNotEmpty()) {
                        val (cx, cy) = stack.removeLast()
                        region.add(Pair(cx, cy))
                        
                        val neighbors = listOf(
                            Pair(cx - 1, cy), Pair(cx + 1, cy),
                            Pair(cx, cy - 1), Pair(cx, cy + 1)
                        )
                        
                        for ((nx, ny) in neighbors) {
                            if (nx in 0 until width && ny in 0 until height) {
                                if (!visited[ny][nx] && bitmap.getPixel(nx, ny) == Color.WHITE) {
                                    visited[ny][nx] = true
                                    stack.add(Pair(nx, ny))
                                }
                            }
                        }
                    }
                    
                    if (region.size > 100) {
                        regions.add(Pair(region, region.size))
                    }
                }
            }
        }
        
        return regions.sortedByDescending { it.second }.take(20)
    }
    
    private fun createMaskFromRegion(width: Int, height: Int, region: Set<Pair<Int, Int>>): Bitmap {
        val mask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(mask)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        
        for ((x, y) in region) {
            mask.setPixel(x, y, Color.WHITE)
        }
        
        return mask
    }
    
    private fun calculateBBox(region: Set<Pair<Int, Int>>, width: Int, height: Int): BoundingBox {
        var minX = width
        var minY = height
        var maxX = 0
        var maxY = 0
        
        for ((x, y) in region) {
            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x)
            maxY = max(maxY, y)
        }
        
        return BoundingBox(minX, minY, maxX - minX, maxY - minY)
    }
    
    private fun classifyRegion(area: Int, bbox: BoundingBox, mask: Bitmap, originalBitmap: Bitmap): String {
        val aspectRatio = if (bbox.height > 0) bbox.width.toFloat() / bbox.height else 1f
        val dominantColor = getDominantColor(originalBitmap, mask)
        val colorName = getColorName(dominantColor)
        
        val position = getPosition(bbox, originalBitmap.width, originalBitmap.height)
        val shape = getShape(aspectRatio, area)
        
        return "$colorName $shape at $position"
    }
    
    private fun getDominantColor(bitmap: Bitmap, mask: Bitmap): Int {
        var r = 0
        var g = 0
        var b = 0
        var count = 0
        
        for (y in 0 until mask.height) {
            for (x in 0 until mask.width) {
                if (mask.getPixel(x, y) != Color.TRANSPARENT && 
                    x < bitmap.width && y < bitmap.height) {
                    val pixel = bitmap.getPixel(x, y)
                    r += Color.red(pixel)
                    g += Color.green(pixel)
                    b += Color.blue(pixel)
                    count++
                }
            }
        }
        
        if (count == 0) return Color.GRAY
        return Color.rgb(r / count, g / count, b / count)
    }
    
    private fun getColorName(color: Int): String {
        val r = Color.red(color)
        val g = Color.green(color)
        val b = Color.blue(color)
        
        val max = maxOf(r, g, b)
        val min = minOf(r, g, b)
        val diff = max - min
        
        if (diff < 30 && max < 80) return "dark"
        if (diff < 30 && max > 180) return "light"
        
        if (g > r && g > b) {
            if (g - maxOf(r, b) > 50) return "green"
        }
        if (r > g && r > b) {
            if (r - maxOf(g, b) > 50) return "red"
        }
        if (b > r && b > g) {
            if (b - maxOf(r, g) > 50) return "blue"
        }
        if (r > 200 && g > 200 && b > 200) return "white"
        if (r < 50 && g < 50 && b < 50) return "black"
        if (r > 150 && g > 100 && b < 100) return "orange"
        if (r > 150 && g > 150 && b < 100) return "yellow"
        if (r > 100 && r < 180 && g > 100 && g < 180 && b > 100 && b < 180) return "gray"
        
        return "colored"
    }
    
    private fun getPosition(bbox: BoundingBox, imgWidth: Int, imgHeight: Int): String {
        val cx = bbox.x + bbox.width / 2
        val cy = bbox.y + bbox.height / 2
        
        val centerX = imgWidth / 2
        val centerY = imgHeight / 2
        
        val dx = cx - centerX
        val dy = cy - centerY
        
        return when {
            abs(dx) < imgWidth / 6 && abs(dy) < imgHeight / 6 -> "center"
            dy < -imgHeight / 6 -> "top"
            dy > imgHeight / 6 -> "bottom"
            dx < -imgWidth / 6 -> "left"
            dx > imgWidth / 6 -> "right"
            else -> "middle"
        }
    }
    
    private fun getShape(aspectRatio: Float, area: Int): String {
        return when {
            area > 80000 -> "large object"
            area > 40000 -> "medium object"
            area > 15000 -> "small object"
            aspectRatio > 2.5f -> "tall object"
            aspectRatio < 0.4f -> "wide object"
            aspectRatio > 1.5f -> "vertical object"
            aspectRatio < 0.67f -> "horizontal object"
            else -> "object"
        }
    }
    
    private fun extractMaskIndices(region: Set<Pair<Int, Int>>, width: Int): List<Int> {
        return region.map { (x, y) -> y * width + x }
    }
    
    private fun generateJsonGraph(objects: List<SegmentedObject>, width: Int, height: Int): String {
        val sb = StringBuilder()
        sb.appendLine("{")
        sb.appendLine("  \"image\": {")
        sb.appendLine("    \"width\": $width,")
        sb.appendLine("    \"height\": $height")
        sb.appendLine("  },")
        sb.appendLine("  \"objects\": [")
        
        objects.forEachIndexed { index, obj ->
            sb.appendLine("    {")
            sb.appendLine("      \"id\": ${obj.id},")
            sb.appendLine("      \"label\": \"${obj.label}\",")
            sb.appendLine("      \"confidence\": ${String.format("%.2f", obj.confidence)},")
            sb.appendLine("      \"bbox\": [${obj.bbox.x}, ${obj.bbox.y}, ${obj.bbox.width}, ${obj.bbox.height}],")
            sb.appendLine("      \"area\": ${obj.maskData.size}")
            sb.append("    }")
            if (index < objects.size - 1) sb.append(",")
            sb.appendLine()
        }
        
        sb.appendLine("  ],")
        sb.appendLine("  \"count\": ${objects.size}")
        sb.appendLine("}")
        
        return sb.toString()
    }
    
    private fun abs(value: Int): Int = if (value < 0) -value else value
}
