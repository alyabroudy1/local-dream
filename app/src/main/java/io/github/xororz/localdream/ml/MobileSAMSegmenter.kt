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

    // Cached encoder results for interactive point-prompt segmentation
    private var cachedEncoderResults: EncoderResults? = null
    private var cachedBitmapHash: Int = 0

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

    /**
     * Interactive point-prompt segmentation.
     * Runs the encoder once (cached) and decoder for the given point.
     * Returns the mask bitmap for the object under the tap point.
     */
    suspend fun segmentAtPoint(bitmap: Bitmap, pointX: Float, pointY: Float): Bitmap? {
        if (!isModelLoaded || encoderSession == null || decoderSession == null) {
            Log.w(TAG, "Model not loaded. Call loadModel() first.")
            return null
        }

        return withContext(Dispatchers.Default) {
            try {
                val bitmapHash = System.identityHashCode(bitmap)

                // Run encoder only if not cached for this bitmap
                if (cachedEncoderResults == null || cachedBitmapHash != bitmapHash) {
                    Log.i(TAG, "Running encoder for new image (${bitmap.width}x${bitmap.height})")
                    val encoderStart = System.currentTimeMillis()
                    cachedEncoderResults = runEncoder(bitmap)
                    cachedBitmapHash = bitmapHash
                    val encoderTime = System.currentTimeMillis() - encoderStart
                    Log.i(TAG, "Encoder completed in ${encoderTime}ms (cached for reuse)")
                } else {
                    Log.i(TAG, "Using cached encoder results")
                }

                val encoderResults = cachedEncoderResults ?: run {
                    Log.e(TAG, "Encoder failed")
                    return@withContext null
                }

                // Run decoder for the single point
                val decoderStart = System.currentTimeMillis()
                val mask = runDecoder(encoderResults, pointX, pointY, bitmap.width, bitmap.height)
                val decoderTime = System.currentTimeMillis() - decoderStart
                Log.i(TAG, "Decoder completed in ${decoderTime}ms for point ($pointX, $pointY)")

                mask
            } catch (e: Exception) {
                Log.e(TAG, "segmentAtPoint failed: ${e.message}")
                e.printStackTrace()
                null
            }
        }
    }

    /**
     * Clear the cached encoder results to free memory.
     */
    fun clearCache() {
        cachedEncoderResults = null
        cachedBitmapHash = 0
        Log.i(TAG, "Encoder cache cleared")
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
            Log.i(TAG, "runEncoder: starting, input ${bitmap.width}x${bitmap.height}")

            // SAM expects longest side = inputDim, aspect ratio preserved, padded to square
            val origW = bitmap.width
            val origH = bitmap.height
            val maxDim = maxOf(origW, origH)
            val scaleToFit = inputDim.toFloat() / maxDim
            val newW = (origW * scaleToFit).toInt()
            val newH = (origH * scaleToFit).toInt()

            val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
            Log.i(TAG, "runEncoder: resized to ${resized.width}x${resized.height} (fit in $inputDim, scale=$scaleToFit)")

            // HWC format: (1024, 1024, 3) - row-major, interleaved RGB, zero-padded
            // NOTE: This ONNX model expects raw pixel values [0, 255] (normalization is baked in)
            val pixels = FloatBuffer.allocate(inputDim * inputDim * 3)
            pixels.rewind()

            var debugCount = 0
            for (y in 0 until inputDim) {
                for (x in 0 until inputDim) {
                    if (x < newW && y < newH) {
                        val pixel = resized.getPixel(x, y)
                        val r = Color.red(pixel).toFloat()
                        val g = Color.green(pixel).toFloat()
                        val b = Color.blue(pixel).toFloat()
                        pixels.put(r)
                        pixels.put(g)
                        pixels.put(b)
                        if (debugCount < 3) {
                            Log.i(TAG, "runEncoder: pixel[$x,$y] RGB=($r, $g, $b)")
                            debugCount++
                        }
                    } else {
                        pixels.put(0f)
                        pixels.put(0f)
                        pixels.put(0f)
                    }
                }
            }
            pixels.rewind()
            Log.i(TAG, "runEncoder: HWC pixel buffer prepared (${inputDim}x${inputDim}x3, raw 0-255)")

            val inputTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                pixels,
                longArrayOf(inputDim.toLong(), inputDim.toLong(), 3)
            )
            Log.i(TAG, "runEncoder: input tensor shape [$inputDim, $inputDim, 3]")

            val outputs = encoderSession!!.run(mapOf(encoderInputName to inputTensor))
            Log.i(TAG, "runEncoder: inference done")

            val embeddingTensor = outputs[imageEmbeddingOutputName].get() as ai.onnxruntime.OnnxTensor
            val imageEmbedding = embeddingTensor.floatBuffer
            val embeddingShape = embeddingTensor.info.shape
            
            Log.i(TAG, "runEncoder: embedding shape=${embeddingShape.toList()}, capacity=${imageEmbedding.capacity()}")

            EncoderResults(imageEmbedding, scaleToFit, newW, newH)
        } catch (e: Exception) {
            Log.e(TAG, "Encoder error: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    private data class EncoderResults(
        val imageEmbedding: FloatBuffer,
        val scaleToFit: Float,    // scale factor from original to encoder space
        val resizedW: Int,        // actual image width in encoder space (before padding)
        val resizedH: Int         // actual image height in encoder space (before padding)
    )


    private fun runDecoder(encoderResults: EncoderResults, pointX: Float, pointY: Float, imgWidth: Int, imgHeight: Int): Bitmap? {
        return try {
            Log.i(TAG, "runDecoder: point=($pointX, $pointY), img=${imgWidth}x${imgHeight}")
            
            val inputs = decoderInputs ?: run {
                Log.e(TAG, "runDecoder: decoderInputs is null")
                return null
            }

            // Transform point from original image space to encoder's 1024x1024 space
            val encoderPointX = pointX * encoderResults.scaleToFit
            val encoderPointY = pointY * encoderResults.scaleToFit
            
            Log.i(TAG, "runDecoder: encoder point=($encoderPointX, $encoderPointY), scale=${encoderResults.scaleToFit}")

            val pointCoords = FloatBuffer.wrap(floatArrayOf(
                encoderPointX, encoderPointY,
                0.0f, 0.0f // Padding point
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

            // orig_im_size should match the encoder input dimensions
            val origSizeTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!,
                FloatBuffer.wrap(floatArrayOf(inputDim.toFloat(), inputDim.toFloat())),
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
            
            // Log all output names for debugging
            val outputNames = outputs.map { it.key }
            Log.i(TAG, "runDecoder: output names: $outputNames")

            // Get mask output
            val maskTensor = outputs[inputs.maskOutputName].get() as ai.onnxruntime.OnnxTensor
            val maskShape = maskTensor.info.shape
            Log.i(TAG, "runDecoder: mask shape=${maskShape.toList()}")

            // Get scores to pick the best mask
            var bestMaskIdx = 0
            if (inputs.scoresOutputName.isNotEmpty()) {
                try {
                    val scoresTensor = outputs[inputs.scoresOutputName].get() as ai.onnxruntime.OnnxTensor
                    val scoresBuffer = scoresTensor.floatBuffer
                    val scoresShape = scoresTensor.info.shape
                    Log.i(TAG, "runDecoder: scores shape=${scoresShape.toList()}")
                    
                    var bestScore = Float.NEGATIVE_INFINITY
                    for (i in 0 until scoresBuffer.capacity()) {
                        val score = scoresBuffer.get(i)
                        Log.i(TAG, "runDecoder: mask[$i] score=$score")
                        if (score > bestScore) {
                            bestScore = score
                            bestMaskIdx = i
                        }
                    }
                    Log.i(TAG, "runDecoder: best mask index=$bestMaskIdx, score=$bestScore")
                } catch (e: Exception) {
                    Log.w(TAG, "runDecoder: could not read scores: ${e.message}")
                }
            }

            val maskBuffer = maskTensor.floatBuffer

            // Determine per-mask dimensions from output shape
            // Output is typically (1, numMasks, H, W) 
            val numMasks: Int
            val maskH: Int
            val maskW: Int
            when (maskShape.size) {
                4 -> { numMasks = maskShape[1].toInt(); maskH = maskShape[2].toInt(); maskW = maskShape[3].toInt() }
                3 -> { numMasks = maskShape[0].toInt(); maskH = maskShape[1].toInt(); maskW = maskShape[2].toInt() }
                2 -> { numMasks = 1; maskH = maskShape[0].toInt(); maskW = maskShape[1].toInt() }
                else -> { numMasks = 1; maskH = inputDim; maskW = inputDim }
            }
            Log.i(TAG, "runDecoder: numMasks=$numMasks, maskDims=${maskW}x${maskH}, bestIdx=$bestMaskIdx")

            // Read the best mask (skip earlier masks)
            val maskOffset = bestMaskIdx * maskH * maskW
            val rawMask = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
            var whiteCount = 0
            for (y in 0 until maskH) {
                for (x in 0 until maskW) {
                    val idx = maskOffset + x + y * maskW
                    if (idx < maskBuffer.capacity() && maskBuffer.get(idx) > 0f) {
                        rawMask.setPixel(x, y, Color.WHITE)
                        whiteCount++
                    }
                }
            }
            val totalPixels = maskW * maskH
            val pct = if (totalPixels > 0) (whiteCount * 100f / totalPixels) else 0f
            Log.i(TAG, "runDecoder: white=$whiteCount/$totalPixels (${String.format("%.1f", pct)}%)")

            // Scale the mask from encoder space to original image dimensions
            val maskBitmap = Bitmap.createScaledBitmap(rawMask, imgWidth, imgHeight, false)

            maskBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Decoder error: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    /**
     * Run decoder with multiple positive and negative points.
     * Positive points (label=1) = include this area
     * Negative points (label=0) = exclude this area
     */
    private fun runDecoderMultiPoint(
        encoderResults: EncoderResults,
        positivePoints: List<Pair<Float, Float>>,
        negativePoints: List<Pair<Float, Float>>,
        imgWidth: Int, imgHeight: Int
    ): Bitmap? {
        return try {
            val inputs = decoderInputs ?: return null
            val totalPoints = positivePoints.size + negativePoints.size + 1 // +1 padding

            Log.i(TAG, "runDecoderMultiPoint: ${positivePoints.size} pos, ${negativePoints.size} neg points")

            val coordsArray = FloatArray(totalPoints * 2)
            val labelsArray = FloatArray(totalPoints)
            var idx = 0

            for (p in positivePoints) {
                val ex = p.first * encoderResults.scaleToFit
                val ey = p.second * encoderResults.scaleToFit
                coordsArray[idx * 2] = ex; coordsArray[idx * 2 + 1] = ey
                labelsArray[idx] = 1.0f
                Log.i(TAG, "  pos[$idx]: (${p.first}, ${p.second}) -> enc ($ex, $ey)")
                idx++
            }
            for (p in negativePoints) {
                val ex = p.first * encoderResults.scaleToFit
                val ey = p.second * encoderResults.scaleToFit
                coordsArray[idx * 2] = ex; coordsArray[idx * 2 + 1] = ey
                labelsArray[idx] = 0.0f
                Log.i(TAG, "  neg[$idx]: (${p.first}, ${p.second}) -> enc ($ex, $ey)")
                idx++
            }
            coordsArray[idx * 2] = 0.0f; coordsArray[idx * 2 + 1] = 0.0f
            labelsArray[idx] = -1.0f

            val embTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!, encoderResults.imageEmbedding, longArrayOf(1, 256, 64, 64))
            val coordsTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!, FloatBuffer.wrap(coordsArray), longArrayOf(1, totalPoints.toLong(), 2))
            val labelsTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!, FloatBuffer.wrap(labelsArray), longArrayOf(1, totalPoints.toLong()))
            val maskInTensor = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!, FloatBuffer.wrap(FloatArray(256 * 256)), longArrayOf(1, 1, 256, 256))
            val hasMaskT = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!, FloatBuffer.wrap(floatArrayOf(0.0f)), longArrayOf(1))
            val origSizeT = ai.onnxruntime.OnnxTensor.createTensor(
                ortEnvironment!!, FloatBuffer.wrap(floatArrayOf(inputDim.toFloat(), inputDim.toFloat())), longArrayOf(2))

            val inputMap = mapOf(
                inputs.imageEmbeddingName to embTensor,
                inputs.pointCoordsName to coordsTensor,
                inputs.pointLabelsName to labelsTensor,
                inputs.maskInputName to maskInTensor,
                inputs.hasMaskInputName to hasMaskT,
                inputs.origImSizeName to origSizeT
            )

            val outputs = decoderSession!!.run(inputMap)
            val maskTensor = outputs[inputs.maskOutputName].get() as ai.onnxruntime.OnnxTensor
            val maskShape = maskTensor.info.shape
            val maskBuffer = maskTensor.floatBuffer

            val numMasks: Int; val maskH: Int; val maskW: Int
            when (maskShape.size) {
                4 -> { numMasks = maskShape[1].toInt(); maskH = maskShape[2].toInt(); maskW = maskShape[3].toInt() }
                3 -> { numMasks = maskShape[0].toInt(); maskH = maskShape[1].toInt(); maskW = maskShape[2].toInt() }
                2 -> { numMasks = 1; maskH = maskShape[0].toInt(); maskW = maskShape[1].toInt() }
                else -> { numMasks = 1; maskH = inputDim; maskW = inputDim }
            }

            var bestMaskIdx = 0
            try {
                val scoresTensor = outputs[inputs.scoresOutputName].get() as ai.onnxruntime.OnnxTensor
                val scoresBuffer = scoresTensor.floatBuffer
                var bestScore = Float.NEGATIVE_INFINITY
                for (i in 0 until scoresBuffer.capacity()) {
                    if (scoresBuffer.get(i) > bestScore) { bestScore = scoresBuffer.get(i); bestMaskIdx = i }
                }
                Log.i(TAG, "runDecoderMultiPoint: best mask=$bestMaskIdx, score=$bestScore")
            } catch (_: Exception) {}

            val maskOffset = bestMaskIdx * maskH * maskW
            val rawMask = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
            var whiteCount = 0
            for (y in 0 until maskH) {
                for (x in 0 until maskW) {
                    val i = maskOffset + x + y * maskW
                    if (i < maskBuffer.capacity() && maskBuffer.get(i) > 0f) {
                        rawMask.setPixel(x, y, Color.WHITE); whiteCount++
                    }
                }
            }
            val pct = if (maskH * maskW > 0) (whiteCount * 100f / (maskH * maskW)) else 0f
            Log.i(TAG, "runDecoderMultiPoint: white=$whiteCount (${String.format("%.1f", pct)}%)")

            Bitmap.createScaledBitmap(rawMask, imgWidth, imgHeight, false)
        } catch (e: Exception) {
            Log.e(TAG, "runDecoderMultiPoint error: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    /**
     * Precise segmentation using positive + negative points.
     * For example, to select just the eyes: positive at eye location,
     * negative at forehead, chin, cheeks to exclude the rest of the face.
     */
    suspend fun segmentPrecise(
        bitmap: Bitmap,
        positivePoints: List<Pair<Float, Float>>,
        negativePoints: List<Pair<Float, Float>>
    ): Bitmap? {
        if (!isModelLoaded) return null
        return withContext(Dispatchers.Default) {
            try {
                val bitmapHash = System.identityHashCode(bitmap)
                if (cachedEncoderResults == null || cachedBitmapHash != bitmapHash) {
                    Log.i(TAG, "segmentPrecise: Running encoder (${bitmap.width}x${bitmap.height})")
                    val t = System.currentTimeMillis()
                    cachedEncoderResults = runEncoder(bitmap)
                    cachedBitmapHash = bitmapHash
                    Log.i(TAG, "segmentPrecise: Encoder done in ${System.currentTimeMillis() - t}ms")
                } else {
                    Log.i(TAG, "segmentPrecise: Using cached encoder")
                }
                val enc = cachedEncoderResults ?: return@withContext null
                val t = System.currentTimeMillis()
                val mask = runDecoderMultiPoint(enc, positivePoints, negativePoints, bitmap.width, bitmap.height)
                Log.i(TAG, "segmentPrecise: Decoder done in ${System.currentTimeMillis() - t}ms")
                mask
            } catch (e: Exception) {
                Log.e(TAG, "segmentPrecise failed: ${e.message}")
                null
            }
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