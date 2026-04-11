package io.github.xororz.localdream.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.set
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer

private fun min(a: Int, b: Int) = if (a < b) a else b
private fun max(a: Int, b: Int) = if (a > b) a else b

class MobileSAMSegmenter(private val context: Context) {

    companion object {
        private const val TAG = "MobileSAMSegmenter"
    }

    private var isModelLoaded = false

    private val encoderPath = "models/mobile_sam/mobile_sam_image_encoder.onnx"
    private val decoderPath = "models/mobile_sam/sam_mask_decoder_single.onnx"

    private var ortEnvironment: ai.onnxruntime.OrtEnvironment? = null
    private var encoderSession: ai.onnxruntime.OrtSession? = null
    private var decoderSession: ai.onnxruntime.OrtSession? = null

    private var encoderInputName: String = ""
    private var decoderInputs: DecoderInputs? = null
    private var imageEmbeddingOutputName: String = ""

    private data class DecoderInputs(
        val imageEmbeddingName: String,
        val pointCoordsName: String,
        val pointLabelsName: String,
        val maskInputName: String,
        val hasMaskInputName: String,
        val origImSizeName: String,
        val maskOutputName: String,
        val scoresOutputName: String
    )

    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val inputDim = 1024

    suspend fun loadModel(): Boolean = withContext(Dispatchers.IO) {
        try {
            val encoderFile = File(context.filesDir, encoderPath)
            val decoderFile = File(context.filesDir, decoderPath)

            if (!encoderFile.exists()) {
                copyAssetToFile(encoderPath, encoderFile)
            }
            if (!decoderFile.exists()) {
                copyAssetToFile(decoderPath, decoderFile)
            }

            ortEnvironment = ai.onnxruntime.OrtEnvironment.getEnvironment()

            val sessionOptions = ai.onnxruntime.OrtSession.SessionOptions().apply {
                setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            encoderSession = ortEnvironment!!.createSession(encoderFile.absolutePath, sessionOptions)
            decoderSession = ortEnvironment!!.createSession(decoderFile.absolutePath, sessionOptions)

            encoderInputName = encoderSession!!.inputNames.first()
            val encoderOutputNames = encoderSession!!.outputNames.toList()
            Log.i(TAG, "Encoder input: $encoderInputName")
            Log.i(TAG, "Encoder outputs: $encoderOutputNames")

            if (encoderOutputNames.isNotEmpty()) {
                imageEmbeddingOutputName = encoderOutputNames[0]
            }

            val decoderInputNames = decoderSession!!.inputNames.toList()
            val decoderOutputNames = decoderSession!!.outputNames.toList()
            Log.i(TAG, "Decoder inputs: $decoderInputNames")
            Log.i(TAG, "Decoder outputs: $decoderOutputNames")

            if (decoderInputNames.size >= 6 && decoderOutputNames.size >= 2) {
                decoderInputs = DecoderInputs(
                    imageEmbeddingName = decoderInputNames[0],
                    pointCoordsName = decoderInputNames[1],
                    pointLabelsName = decoderInputNames[2],
                    maskInputName = decoderInputNames[3],
                    hasMaskInputName = decoderInputNames[4],
                    origImSizeName = decoderInputNames[5],
                    maskOutputName = decoderOutputNames[0],
                    scoresOutputName = decoderOutputNames[1]
                )
            }

            isModelLoaded = true
            Log.i(TAG, "MobileSAM models loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}")
            e.printStackTrace()
            isModelLoaded = false
            false
        }
    }

    private fun copyAssetToFile(assetPath: String, destFile: File) {
        context.assets.open(assetPath).use { input ->
            destFile.parentFile?.mkdirs()
            destFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        Log.i(TAG, "Copied $assetPath to ${destFile.absolutePath}")
    }

    suspend fun segmentImage(bitmap: Bitmap, numPoints: Int = 32): ImageObjects? {
        if (!isModelLoaded || encoderSession == null || decoderSession == null) {
            Log.w(TAG, "Model not loaded. Call loadModel() first.")
            return null
        }

        return withContext(Dispatchers.Default) {
            try {
                Log.i(TAG, "Running encoder on ${bitmap.width}x${bitmap.height} image")
                val encoderStart = System.currentTimeMillis()
                val encoderResults = runEncoder(bitmap)
                val encoderTime = System.currentTimeMillis() - encoderStart
                if (encoderResults == null) {
                    Log.e(TAG, "Encoder failed")
                    return@withContext null
                }
                Log.i(TAG, "Encoder completed in ${encoderTime}ms, running decoder for $numPoints points")

                val points = generateGridPoints(numPoints, bitmap.width, bitmap.height)
                val objects = mutableListOf<SegmentedObject>()
                var objectId = 0
                val decoderStart = System.currentTimeMillis()

                for ((idx, point) in points.withIndex()) {
                    if (objects.size >= 20) {
                        Log.i(TAG, "Reached maximum of 20 distinct objects, stopping inference.")
                        break
                    }

                    Log.i(TAG, "Processing point ${idx + 1}/${points.size} at (${point.first}, ${point.second})")
                    val inferStart = System.currentTimeMillis()
                    val mask = runDecoder(encoderResults, point.first, point.second, bitmap.width, bitmap.height)
                    val inferTime = System.currentTimeMillis() - inferStart
                    
                    val area = if (mask != null) calculateMaskArea(mask) else 0
                    Log.i(TAG, "Point ${idx + 1} yielded mask of area $area in ${inferTime}ms")
                    
                    if (area > 500) {
                        val bbox = calculateBBox(mask!!)
                        val maskData = mask.let { extractMaskDataCompact(it) }
                        
                        var isDuplicate = false
                        var maxIou = 0f
                        for (existingObj in objects) {
                            val iou = calculateIoU(existingObj.maskData, existingObj.bbox, maskData, bbox)
                            if (iou > maxIou) maxIou = iou
                            if (iou > 0.85f) {
                                isDuplicate = true
                                break
                            }
                        }
                        
                        if (isDuplicate) {
                            Log.i(TAG, String.format("Discarding mask from point %d due to high overlap with existing object (IoU: %.2f)", idx + 1, maxIou))
                            continue
                        }

                        val label = classifyRegionWithColor(bitmap, mask)
                        
                        objects.add(
                            SegmentedObject(
                                id = objectId++,
                                label = label,
                                confidence = 0.9f,
                                bbox = bbox,
                                mask = mask,
                                maskData = maskData
                            )
                        )
                        Log.i(TAG, "Added new distinct object #$objectId: label='$label', bbox=[${bbox.x}, ${bbox.y}, ${bbox.width}x${bbox.height}]")
                    } else {
                        Log.i(TAG, "Discarding mask from point ${idx + 1} due to small area ($area pixels)")
                    }
                }
                
                val decoderTime = System.currentTimeMillis() - decoderStart
                Log.i(TAG, "Decoder loop completed in ${decoderTime}ms, found ${objects.size} distinct objects, releasing decoder outputs")

                if (objects.isEmpty()) {
                    val fullMask = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                    val canvas = Canvas(fullMask)
                    canvas.drawColor(Color.WHITE)
                    objects.add(
                        SegmentedObject(
                            id = 0,
                            label = "full_image",
                            confidence = 1.0f,
                            bbox = BoundingBox(0, 0, bitmap.width, bitmap.height),
                            mask = fullMask,
                            maskData = extractMaskData(fullMask)
                        )
                    )
                }

                val json = generateJsonGraph(objects, bitmap.width, bitmap.height)
                Log.i(TAG, "Found ${objects.size} objects")

                ImageObjects(
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    objects = objects,
                    json = json
                )
            } catch (e: Exception) {
                Log.e(TAG, "Segmentation failed: ${e.message}")
                e.printStackTrace()
                null
            }
        }
    }

    private fun runEncoder(bitmap: Bitmap): EncoderResults? {
        return try {
            Log.i(TAG, "runEncoder: starting")
            
            val resized = Bitmap.createScaledBitmap(bitmap, inputDim, inputDim, true)
            Log.i(TAG, "runEncoder: resized to ${resized.width}x${resized.height}")

            val pixels = FloatBuffer.allocate(1 * 3 * inputDim * inputDim)
            pixels.rewind()

            for (i in 0 until inputDim) {
                for (j in 0 until inputDim) {
                    val pixel = resized[j, i]
                    val r = ((Color.red(pixel) / 255.0f) - mean[0]) / std[0]
                    val g = ((Color.green(pixel) / 255.0f) - mean[1]) / std[1]
                    val b = ((Color.blue(pixel) / 255.0f) - mean[2]) / std[2]
                    pixels.put(r)
                    pixels.put(g)
                    pixels.put(b)
                }
            }
            pixels.rewind()
            Log.i(TAG, "runEncoder: pixel buffer prepared")

            val inputTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                pixels,
                longArrayOf(inputDim.toLong(), inputDim.toLong(), 3)
            )
            Log.i(TAG, "runEncoder: input tensor created")

            val outputs = encoderSession!!.run(mapOf(encoderInputName to inputTensor))
            Log.i(TAG, "runEncoder: inference done")

            val imageEmbedding = (outputs[imageEmbeddingOutputName].get() as ai.onnxruntime.OnnxTensor).floatBuffer
            
            Log.i(TAG, "runEncoder: imageEmbedding capacity: ${imageEmbedding.capacity()}")

            EncoderResults(imageEmbedding)
        } catch (e: Exception) {
            Log.e(TAG, "Encoder error: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    private data class EncoderResults(
        val imageEmbedding: FloatBuffer
    )

    private fun runDecoder(encoderResults: EncoderResults, pointX: Float, pointY: Float, imgWidth: Int, imgHeight: Int): Bitmap? {
        return try {
            Log.i(TAG, "runDecoder: point=($pointX, $pointY), img=${imgWidth}x${imgHeight}")
            
            val inputs = decoderInputs ?: run {
                Log.e(TAG, "runDecoder: decoderInputs is null")
                return null
            }

            val scaleX = imgWidth.toFloat() / inputDim
            val scaleY = imgHeight.toFloat() / inputDim

            val pointCoords = FloatBuffer.wrap(floatArrayOf(
                pointX / scaleX, pointY / scaleY,
                0.0f, 0.0f // Padding point required by SAM ONNX for "no bounding box" queries
            ))
            val pointLabels = FloatBuffer.wrap(floatArrayOf(
                1.0f,  // Foreground positive point
                -1.0f  // Padding point label
            ))

            val embeddingTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                encoderResults.imageEmbedding,
                longArrayOf(1, 256, 64, 64)
            )

            val pointCoordsTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                pointCoords,
                longArrayOf(1, 2, 2)
            )

            val pointLabelsTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                pointLabels,
                longArrayOf(1, 2)
            )

            val emptyMask = FloatBuffer.wrap(FloatArray(1 * 256 * 256) { 0f })
            val maskInputTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                emptyMask,
                longArrayOf(1, 1, 256, 256)
            )

            val hasMaskTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                FloatBuffer.wrap(floatArrayOf(0.0f)),
                longArrayOf(1)
            )

            val origSizeTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                FloatBuffer.wrap(floatArrayOf(imgHeight.toFloat(), imgWidth.toFloat())),
                longArrayOf(2)
            )

            val inputMap = mapOf(
                inputs.imageEmbeddingName to embeddingTensor,
                inputs.pointCoordsName to pointCoordsTensor,
                inputs.pointLabelsName to pointLabelsTensor,
                inputs.maskInputName to maskInputTensor,
                inputs.hasMaskInputName to hasMaskTensor,
                inputs.origImSizeName to origSizeTensor
            )
            
            Log.i(TAG, "runDecoder: running inference...")
            val outputs = decoderSession!!.run(inputMap)
            Log.i(TAG, "runDecoder: inference done")

            val maskBuffer = (outputs[inputs.maskOutputName].get() as ai.onnxruntime.OnnxTensor).floatBuffer
            Log.i(TAG, "runDecoder: mask capacity: ${maskBuffer.capacity()}")

            val maskBitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888)
            var whiteCount = 0
            for (i in 0 until imgHeight) {
                for (j in 0 until imgWidth) {
                    val idx = j + i * imgWidth
                    if (idx < maskBuffer.capacity() && maskBuffer.get(idx) > 0f) {
                        maskBitmap[j, i] = Color.WHITE
                        whiteCount++
                    } else {
                        maskBitmap[j, i] = Color.TRANSPARENT
                    }
                }
            }

            Log.i(TAG, "runDecoder: white pixels: $whiteCount")
            maskBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Decoder error: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    private fun generateGridPoints(numPoints: Int, width: Int, height: Int): List<Pair<Float, Float>> {
        val points = mutableListOf<Pair<Float, Float>>()
        val rows = kotlin.math.sqrt(numPoints.toDouble()).toInt()
        val cols = (numPoints + rows - 1) / rows

        val stepX = width.toFloat() / (cols + 1)
        val stepY = height.toFloat() / (rows + 1)

        for (i in 1..rows) {
            for (j in 1..cols) {
                if (points.size >= numPoints) break
                points.add(Pair(j * stepX, i * stepY))
            }
        }

        points.add(Pair(width / 2f, height / 2f))
        return points
    }

    private fun calculateMaskArea(mask: Bitmap): Int {
        var count = 0
        for (x in 0 until mask.width) {
            for (y in 0 until mask.height) {
                if (mask.getPixel(x, y) != Color.TRANSPARENT) count++
            }
        }
        return count
    }

    private fun calculateBBox(mask: Bitmap): BoundingBox {
        var minX = mask.width
        var minY = mask.height
        var maxX = 0
        var maxY = 0

        for (x in 0 until mask.width) {
            for (y in 0 until mask.height) {
                if (mask.getPixel(x, y) != Color.TRANSPARENT) {
                    minX = minOf(minX, x)
                    minY = minOf(minY, y)
                    maxX = maxOf(maxX, x)
                    maxY = maxOf(maxY, y)
                }
            }
        }

        if (minX >= maxX || minY >= maxY) {
            return BoundingBox(0, 0, mask.width, mask.height)
        }

        return BoundingBox(minX, minY, maxX - minX, maxY - minY)
    }

    private fun extractMaskData(mask: Bitmap): List<Int> {
        val data = mutableListOf<Int>()
        for (y in 0 until mask.height) {
            for (x in 0 until mask.width) {
                if (mask.getPixel(x, y) != Color.TRANSPARENT) {
                    data.add(y * mask.width + x)
                }
            }
        }
        return data
    }

    private fun extractMaskDataCompact(mask: Bitmap): List<Int> {
        val area = calculateMaskArea(mask)
        if (area > 50000) {
            return listOf(-1)
        }
        return extractMaskData(mask)
    }

    private fun classifyRegion(area: Int): String {
        return when {
            area > 100000 -> "large_object"
            area > 50000 -> "medium_object"
            area > 20000 -> "small_object"
            else -> "detail"
        }
    }
    
    private fun classifyRegionWithColor(bitmap: Bitmap, mask: Bitmap): String {
        val area = calculateMaskArea(mask)
        val bbox = calculateBBox(mask)
        
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
        
        val colorName = if (count > 0) {
            val avgR = r / count
            val avgG = g / count
            val avgB = b / count
            getColorNameFromRGB(avgR, avgG, avgB)
        } else {
            "gray"
        }
        
        val sizeDesc = when {
            area > 80000 -> "large"
            area > 40000 -> "medium"
            area > 15000 -> "small"
            else -> "tiny"
        }
        
        val aspectRatio = if (bbox.height > 0) bbox.width.toFloat() / bbox.height else 1f
        val shapeDesc = when {
            aspectRatio > 2f -> "tall"
            aspectRatio < 0.5f -> "wide"
            else -> ""
        }
        
        val centerX = bbox.x + bbox.width / 2
        val centerY = bbox.y + bbox.height / 2
        
        val hPos = when {
            centerX < bitmap.width / 3f -> "Left"
            centerX > 2 * bitmap.width / 3f -> "Right"
            else -> "Center"
        }
        
        val vPos = when {
            centerY < bitmap.height / 3f -> "Top"
            centerY > 2 * bitmap.height / 3f -> "Bottom"
            else -> "Middle"
        }
        
        val posDesc = if (vPos == "Middle" && hPos == "Center") "Center" else "$vPos $hPos"
        
        return "$colorName $sizeDesc $shapeDesc object at $posDesc".trim().replace("  ", " ")
    }
    
    private fun calculateIoU(mask1Data: List<Int>, bbox1: BoundingBox, mask2Data: List<Int>, bbox2: BoundingBox): Float {
        // If either mask is compact encoded [-1], fall back to Bounding Box IoU
        if (mask1Data.firstOrNull() == -1 || mask2Data.firstOrNull() == -1) {
            val intersectLeft = maxOf(bbox1.x, bbox2.x)
            val intersectTop = maxOf(bbox1.y, bbox2.y)
            val intersectRight = minOf(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
            val intersectBottom = minOf(bbox1.y + bbox1.height, bbox2.y + bbox2.height)

            if (intersectLeft < intersectRight && intersectTop < intersectBottom) {
                val intersectionArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop)
                val unionArea = (bbox1.width * bbox1.height) + (bbox2.width * bbox2.height) - intersectionArea
                if (unionArea <= 0) return 0f
                return intersectionArea.toFloat() / unionArea.toFloat()
            }
            return 0f
        }

        // Precise Pixel IoU
        val set1 = mask1Data.toSet()
        var intersectionSize = 0
        for (pixel in mask2Data) {
            if (set1.contains(pixel)) {
                intersectionSize++
            }
        }
        val unionSize = mask1Data.size + mask2Data.size - intersectionSize
        if (unionSize == 0) return 0f
        return intersectionSize.toFloat() / unionSize.toFloat()
    }
    
    private fun getColorNameFromRGB(r: Int, g: Int, b: Int): String {
        val max = maxOf(r, g, b)
        val min = minOf(r, g, b)
        val diff = max - min
        
        if (diff < 25 && max < 70) return "dark"
        if (diff < 25 && max > 200) return "light"
        
        if (g > r + 30 && g > b + 30) return "green"
        if (r > g + 30 && r > b + 30) return "red"
        if (b > r + 30 && b > g + 30) return "blue"
        if (r > 220 && g > 220 && b > 220) return "white"
        if (r < 40 && g < 40 && b < 40) return "black"
        if (r > 180 && g > 120 && b < 100) return "orange"
        if (r > 200 && g > 180 && b < 80) return "yellow"
        if (r in 80..180 && g in 80..180 && b in 80..180) return "gray"
        
        return "colored"
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

    fun close() {
        isModelLoaded = false
        encoderSession?.close()
        decoderSession?.close()
        ortEnvironment?.close()
    }
}