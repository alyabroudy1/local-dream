package io.github.xororz.localdream.ml

import android.content.Context
import android.graphics.*
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.*

/**
 * Face Swap Engine using InsightFace pipeline:
 *   1. SCRFD face detection (bboxes + 5-point landmarks)
 *   2. ArcFace embedding (512d identity vector)
 *   3. inswapper_128 face swap
 *
 * Models (~725MB total) are downloaded at runtime to context.filesDir/faceswap/
 */
class FaceSwapEngine(private val context: Context) {

    companion object {
        private const val TAG = "FaceSwapEngine"

        // Model file names (stored in context.filesDir/faceswap/)
        const val DETECTOR_MODEL = "det_10g.onnx"        // SCRFD ~16MB
        const val RECOGNIZER_MODEL = "w600k_r50.onnx"    // ArcFace ~167MB
        const val SWAPPER_MODEL = "inswapper_128.onnx"   // inswapper ~555MB

        // Model download URLs (HuggingFace)
        val MODEL_URLS = mapOf(
            DETECTOR_MODEL to "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
            RECOGNIZER_MODEL to "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
            SWAPPER_MODEL to "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        )

        // Standard 5-point alignment template for ArcFace (112x112)
        private val ARCFACE_DST = floatArrayOf(
            38.2946f, 51.6963f,  // left eye
            73.5318f, 51.5014f,  // right eye
            56.0252f, 71.7366f,  // nose
            41.5493f, 92.3655f,  // left mouth
            70.7299f, 92.2041f   // right mouth
        )

        // Standard 5-point alignment template for inswapper (128x128)
        private val INSWAPPER_DST = floatArrayOf(
            43.7655f, 59.0875f,  // left eye  (scaled from 112 template)
            84.0364f, 58.8588f,  // right eye
            64.0288f, 82.0135f,  // nose
            47.4849f, 105.5601f, // left mouth
            80.8342f, 105.3762f  // right mouth
        )

        fun getModelDir(context: Context): File {
            return File(context.filesDir, "faceswap").also { it.mkdirs() }
        }

        fun areModelsDownloaded(context: Context): Boolean {
            val dir = getModelDir(context)
            return listOf(DETECTOR_MODEL, RECOGNIZER_MODEL, SWAPPER_MODEL).all {
                File(dir, it).exists()
            }
        }
    }

    data class DetectedFace(
        val bbox: RectF,
        val landmarks: FloatArray, // 5 points × 2 coords = 10 floats
        val score: Float,
        var embedding: FloatArray? = null // 512d ArcFace embedding
    )

    private var ortEnv: OrtEnvironment? = null
    private var detectorSession: OrtSession? = null
    private var recognizerSession: OrtSession? = null
    private var swapperSession: OrtSession? = null
    private var emapMatrix: FloatArray? = null // emap from inswapper model
    private var isLoaded = false

    /**
     * Load all three ONNX models.
     */
    suspend fun loadModels(): Boolean = withContext(Dispatchers.IO) {
        try {
            val modelDir = getModelDir(context)

            val detFile = File(modelDir, DETECTOR_MODEL)
            val recFile = File(modelDir, RECOGNIZER_MODEL)
            val swapFile = File(modelDir, SWAPPER_MODEL)

            if (!detFile.exists() || !recFile.exists() || !swapFile.exists()) {
                Log.e(TAG, "Models not found in ${modelDir.absolutePath}")
                return@withContext false
            }

            ortEnv = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
            }

            Log.i(TAG, "Loading detector model...")
            detectorSession = ortEnv!!.createSession(detFile.absolutePath, opts)
            Log.i(TAG, "Detector loaded. Inputs: ${detectorSession!!.inputNames}, Outputs: ${detectorSession!!.outputNames}")

            Log.i(TAG, "Loading recognizer model...")
            recognizerSession = ortEnv!!.createSession(recFile.absolutePath, opts)
            Log.i(TAG, "Recognizer loaded. Inputs: ${recognizerSession!!.inputNames}, Outputs: ${recognizerSession!!.outputNames}")

            Log.i(TAG, "Loading swapper model...")
            swapperSession = ortEnv!!.createSession(swapFile.absolutePath, opts)
            Log.i(TAG, "Swapper loaded. Inputs: ${swapperSession!!.inputNames}, Outputs: ${swapperSession!!.outputNames}")

            // Extract emap matrix from model
            Log.i(TAG, "Extracting emap matrix from swapper model...")
            emapMatrix = extractEmapFromOnnx(swapFile)
            if (emapMatrix != null) {
                Log.i(TAG, "Emap extracted: ${emapMatrix!!.size} floats (${sqrt(emapMatrix!!.size.toFloat()).toInt()}x${sqrt(emapMatrix!!.size.toFloat()).toInt()})")
            } else {
                Log.w(TAG, "Failed to extract emap - face swap quality will be degraded")
            }

            isLoaded = true
            Log.i(TAG, "All face swap models loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}", e)
            isLoaded = false
            false
        }
    }

    /**
     * Release all model resources.
     */
    fun release() {
        detectorSession?.close()
        recognizerSession?.close()
        swapperSession?.close()
        detectorSession = null
        recognizerSession = null
        swapperSession = null
        isLoaded = false
        Log.i(TAG, "Models released")
    }

    /**
     * Full face swap pipeline:
     * 1. Detect faces in source → extract embedding
     * 2. Detect faces in target
     * 3. Swap each target face with source identity
     */
    suspend fun swapFaces(
        sourceImage: Bitmap,
        targetImage: Bitmap,
        onProgress: ((String) -> Unit)? = null
    ): Bitmap? = withContext(Dispatchers.Default) {
        if (!isLoaded) {
            Log.e(TAG, "Models not loaded")
            return@withContext null
        }

        try {
            // 1. Detect source face
            onProgress?.invoke("Detecting source face...")
            val sourceFaces = detectFaces(sourceImage)
            if (sourceFaces.isEmpty()) {
                Log.e(TAG, "No face found in source image")
                return@withContext null
            }
            val sourceFace = sourceFaces[0] // Use first/largest face
            Log.i(TAG, "Source face detected: bbox=${sourceFace.bbox}, score=${sourceFace.score}")

            // 2. Extract source embedding
            onProgress?.invoke("Extracting face identity...")
            sourceFace.embedding = extractEmbedding(sourceImage, sourceFace)
            if (sourceFace.embedding == null) {
                Log.e(TAG, "Failed to extract source embedding")
                return@withContext null
            }
            Log.i(TAG, "Source embedding extracted: dim=${sourceFace.embedding!!.size}")

            // 3. Detect target face(s)
            onProgress?.invoke("Detecting target face...")
            val targetFaces = detectFaces(targetImage)
            if (targetFaces.isEmpty()) {
                Log.e(TAG, "No face found in target image")
                return@withContext null
            }
            Log.i(TAG, "Found ${targetFaces.size} face(s) in target")

            // 4. Swap each target face
            onProgress?.invoke("Swapping face...")
            var result = targetImage.copy(Bitmap.Config.ARGB_8888, true)
            for ((idx, targetFace) in targetFaces.withIndex()) {
                Log.i(TAG, "Swapping face $idx: bbox=${targetFace.bbox}")
                result = swapSingleFace(result, targetFace, sourceFace.embedding!!) ?: result
            }

            Log.i(TAG, "Face swap complete: ${result.width}x${result.height}")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Face swap failed: ${e.message}", e)
            null
        }
    }

    // ─── SCRFD Face Detection ───────────────────────────────────────────

    private fun detectFaces(image: Bitmap): List<DetectedFace> {
        val session = detectorSession ?: return emptyList()
        val inputSize = 640

        // Resize image to 640x640
        val ratio = minOf(inputSize.toFloat() / image.width, inputSize.toFloat() / image.height)
        val newW = (image.width * ratio).toInt()
        val newH = (image.height * ratio).toInt()
        val resized = Bitmap.createScaledBitmap(image, newW, newH, true)

        // Pad to 640x640
        val padded = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawBitmap(resized, 0f, 0f, null)

        // Convert to float tensor [1, 3, 640, 640] with mean subtraction
        val inputData = FloatArray(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        padded.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF).toFloat() - 127.5f
            val g = ((pixel shr 8) and 0xFF).toFloat() - 127.5f
            val b = (pixel and 0xFF).toFloat() - 127.5f
            // BGR order for SCRFD
            inputData[i] = b / 128f
            inputData[inputSize * inputSize + i] = g / 128f
            inputData[2 * inputSize * inputSize + i] = r / 128f
        }

        val inputTensor = OnnxTensor.createTensor(
            ortEnv!!, FloatBuffer.wrap(inputData), longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val outputs = session.run(mapOf(session.inputNames.first() to inputTensor))
        val outputNames = outputs.map { it.key }
        Log.i(TAG, "SCRFD outputs: $outputNames")

        // Parse SCRFD outputs: stride 8/16/32, each has scores + bboxes + landmarks
        val faces = mutableListOf<DetectedFace>()
        val strides = listOf(8, 16, 32)
        val scoreThreshold = 0.5f

        try {
            for ((strideIdx, stride) in strides.withIndex()) {
                val scoresKey = outputNames.getOrNull(strideIdx) ?: continue
                val bboxKey = outputNames.getOrNull(strideIdx + 3) ?: continue
                val landmarkKey = outputNames.getOrNull(strideIdx + 6) ?: continue

                val scoresTensor = outputs[scoresKey].get() as OnnxTensor
                val bboxTensor = outputs[bboxKey].get() as OnnxTensor
                val landmarkTensor = outputs[landmarkKey].get() as OnnxTensor

                val scoresData = scoresTensor.floatBuffer
                val bboxData = bboxTensor.floatBuffer
                val landmarkData = landmarkTensor.floatBuffer

                val gridH = inputSize / stride
                val gridW = inputSize / stride
                val numAnchors = 2

                for (h in 0 until gridH) {
                    for (w in 0 until gridW) {
                        for (a in 0 until numAnchors) {
                            val anchorIdx = (h * gridW + w) * numAnchors + a
                            val score = scoresData.get(anchorIdx)

                            if (score > scoreThreshold) {
                                val baseIdx = anchorIdx * 4
                                val cx = (w + 0.5f) * stride
                                val cy = (h + 0.5f) * stride
                                val x1 = (cx - bboxData.get(baseIdx) * stride) / ratio
                                val y1 = (cy - bboxData.get(baseIdx + 1) * stride) / ratio
                                val x2 = (cx + bboxData.get(baseIdx + 2) * stride) / ratio
                                val y2 = (cy + bboxData.get(baseIdx + 3) * stride) / ratio

                                val lmkBaseIdx = anchorIdx * 10
                                val landmarks = FloatArray(10)
                                for (li in 0 until 5) {
                                    landmarks[li * 2] = (cx + landmarkData.get(lmkBaseIdx + li * 2) * stride) / ratio
                                    landmarks[li * 2 + 1] = (cy + landmarkData.get(lmkBaseIdx + li * 2 + 1) * stride) / ratio
                                }

                                faces.add(DetectedFace(
                                    bbox = RectF(x1, y1, x2, y2),
                                    landmarks = landmarks,
                                    score = score
                                ))
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "SCRFD parsing error (trying simplified approach): ${e.message}")
            // Some SCRFD exports have different output format, try simpler parsing
            return parseSimpleSCRFD(outputs, outputNames, ratio)
        }

        // NMS
        val nmsResult = nms(faces, 0.4f)
        Log.i(TAG, "SCRFD: detected ${faces.size} → ${nmsResult.size} after NMS")
        return nmsResult
    }

    private fun parseSimpleSCRFD(
        outputs: OrtSession.Result,
        outputNames: List<String>,
        ratio: Float
    ): List<DetectedFace> {
        // Fallback: try to parse outputs with different naming conventions
        val faces = mutableListOf<DetectedFace>()
        Log.i(TAG, "Attempting simplified SCRFD parsing with ${outputNames.size} outputs")

        // Log tensor shapes for debugging
        for (name in outputNames) {
            try {
                val tensor = outputs[name].get() as OnnxTensor
                Log.i(TAG, "Output '$name': shape=${tensor.info.shape.toList()}")
            } catch (e: Exception) {
                Log.w(TAG, "Could not read output '$name': ${e.message}")
            }
        }

        return faces
    }

    // ─── ArcFace Embedding ──────────────────────────────────────────────

    private fun extractEmbedding(image: Bitmap, face: DetectedFace): FloatArray? {
        val session = recognizerSession ?: return null

        // Align face to 112×112 using 5-point landmarks
        val aligned = alignFace(image, face.landmarks, ARCFACE_DST, 112)
            ?: return null

        // Convert to float tensor [1, 3, 112, 112] normalized to [-1, 1]
        val inputSize = 112
        val inputData = FloatArray(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        aligned.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF).toFloat() / 127.5f - 1f
            val g = ((pixel shr 8) and 0xFF).toFloat() / 127.5f - 1f
            val b = (pixel and 0xFF).toFloat() / 127.5f - 1f
            // RGB order for ArcFace
            inputData[i] = r
            inputData[inputSize * inputSize + i] = g
            inputData[2 * inputSize * inputSize + i] = b
        }

        val inputTensor = OnnxTensor.createTensor(
            ortEnv!!, FloatBuffer.wrap(inputData), longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val outputs = session.run(mapOf(session.inputNames.first() to inputTensor))
        val outputNames = outputs.map { it.key }
        val embeddingTensor = outputs[outputNames[0]].get() as OnnxTensor
        val embedding = embeddingTensor.floatBuffer
        val result = FloatArray(embedding.capacity())
        embedding.get(result)

        // L2 normalize
        val norm = sqrt(result.sumOf { (it * it).toDouble() }).toFloat()
        if (norm > 0) {
            for (i in result.indices) result[i] /= norm
        }

        aligned.recycle()
        Log.i(TAG, "ArcFace embedding: dim=${result.size}, norm=$norm")
        return result
    }

    // ─── inswapper_128 Face Swap ────────────────────────────────────────

    private fun swapSingleFace(
        targetImage: Bitmap,
        targetFace: DetectedFace,
        sourceEmbedding: FloatArray
    ): Bitmap? {
        val session = swapperSession ?: return null

        // Align target face to 128×128
        val aligned = alignFace(targetImage, targetFace.landmarks, INSWAPPER_DST, 128)
            ?: return null

        // Prepare target face tensor [1, 3, 128, 128] - RGB order, [0, 1] range
        val inputSize = 128
        val inputData = FloatArray(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        aligned.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF).toFloat() / 255f
            val g = ((pixel shr 8) and 0xFF).toFloat() / 255f
            val b = (pixel and 0xFF).toFloat() / 255f
            // RGB order for inswapper
            inputData[i] = r
            inputData[inputSize * inputSize + i] = g
            inputData[2 * inputSize * inputSize + i] = b
        }

        val targetTensor = OnnxTensor.createTensor(
            ortEnv!!, FloatBuffer.wrap(inputData), longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        // Transform source embedding through emap matrix
        val transformedEmbedding = if (emapMatrix != null) {
            val emap = emapMatrix!!
            val dim = sourceEmbedding.size // 512
            val result = FloatArray(dim)
            // Matrix multiply: result = sourceEmbedding × emap (emap is [512, 512])
            for (j in 0 until dim) {
                var sum = 0f
                for (k in 0 until dim) {
                    sum += sourceEmbedding[k] * emap[k * dim + j]
                }
                result[j] = sum
            }
            // L2 normalize the result
            val norm = sqrt(result.sumOf { (it * it).toDouble() }).toFloat()
            if (norm > 0) for (i in result.indices) result[i] /= norm
            Log.i(TAG, "Embedding transformed through emap, norm=$norm")
            result
        } else {
            // Fallback: use raw embedding (quality will be poor)
            Log.w(TAG, "Using raw embedding without emap transformation")
            sourceEmbedding
        }

        // Prepare source embedding tensor [1, 512]
        val embeddingTensor = OnnxTensor.createTensor(
            ortEnv!!, FloatBuffer.wrap(transformedEmbedding), longArrayOf(1, transformedEmbedding.size.toLong())
        )

        // Run swapper
        val inputNames = session.inputNames.toList()
        val inputMap = mutableMapOf<String, OnnxTensor>()
        if (inputNames.size >= 2) {
            inputMap[inputNames[0]] = targetTensor
            inputMap[inputNames[1]] = embeddingTensor
        }

        val outputs = session.run(inputMap)
        val swapOutputNames = outputs.map { it.key }
        val outputTensor = outputs[swapOutputNames[0]].get() as OnnxTensor
        val outputShape = outputTensor.info.shape
        Log.i(TAG, "Swapper output shape: ${outputShape.toList()}")

        val outputData = outputTensor.floatBuffer
        val swappedFace = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val outPixels = IntArray(inputSize * inputSize)

        for (i in 0 until inputSize * inputSize) {
            // Output is RGB, [0, 1] range
            val r = (outputData.get(i) * 255f).toInt().coerceIn(0, 255)
            val g = (outputData.get(inputSize * inputSize + i) * 255f).toInt().coerceIn(0, 255)
            val b = (outputData.get(2 * inputSize * inputSize + i) * 255f).toInt().coerceIn(0, 255)
            outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        swappedFace.setPixels(outPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Paste swapped face back using inverse affine transform + blending
        val result = pasteBack(targetImage, swappedFace, targetFace.landmarks)

        aligned.recycle()
        swappedFace.recycle()
        return result
    }

    // ─── Face Alignment (Affine Transform) ──────────────────────────────

    /**
     * Align face using 5-point landmarks → affine transform to target template.
     */
    private fun alignFace(
        image: Bitmap,
        srcLandmarks: FloatArray, // 10 floats (5 points × 2)
        dstTemplate: FloatArray,  // 10 floats (5 points × 2)
        outputSize: Int
    ): Bitmap? {
        try {
            // Use first 3 points (left eye, right eye, nose) for affine estimation
            val srcPoints = floatArrayOf(
                srcLandmarks[0], srcLandmarks[1],
                srcLandmarks[2], srcLandmarks[3],
                srcLandmarks[4], srcLandmarks[5]
            )
            val dstPoints = floatArrayOf(
                dstTemplate[0], dstTemplate[1],
                dstTemplate[2], dstTemplate[3],
                dstTemplate[4], dstTemplate[5]
            )

            // Compute affine transform matrix
            val matrix = computeAffineTransform(srcPoints, dstPoints)

            // Apply transform
            val result = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(result)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
            canvas.drawBitmap(image, matrix, paint)

            return result
        } catch (e: Exception) {
            Log.e(TAG, "Face alignment failed: ${e.message}", e)
            return null
        }
    }

    /**
     * Compute affine transformation matrix from 3 source points to 3 destination points.
     */
    private fun computeAffineTransform(src: FloatArray, dst: FloatArray): Matrix {
        val matrix = Matrix()
        matrix.setPolyToPoly(src, 0, dst, 0, 3)
        return matrix
    }

    /**
     * Paste the swapped 128x128 face back onto the original image.
     * Uses inverse affine transform + elliptical mask for smooth blending.
     */
    private fun pasteBack(
        original: Bitmap,
        swappedFace: Bitmap,
        landmarks: FloatArray
    ): Bitmap {
        val result = original.copy(Bitmap.Config.ARGB_8888, true)

        // Compute the forward affine (src → 128x128)
        val srcPoints = floatArrayOf(
            landmarks[0], landmarks[1],
            landmarks[2], landmarks[3],
            landmarks[4], landmarks[5]
        )
        val dstPoints = floatArrayOf(
            INSWAPPER_DST[0], INSWAPPER_DST[1],
            INSWAPPER_DST[2], INSWAPPER_DST[3],
            INSWAPPER_DST[4], INSWAPPER_DST[5]
        )

        // Compute inverse affine (128x128 → original space)
        val forwardMatrix = Matrix()
        forwardMatrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)
        val inverseMatrix = Matrix()
        if (!forwardMatrix.invert(inverseMatrix)) {
            Log.e(TAG, "Cannot invert affine matrix, falling back to direct paste")
            return result
        }

        // Create an elliptical mask for smooth blending (avoid sharp edges)
        val maskSize = 128
        val mask = Bitmap.createBitmap(maskSize, maskSize, Bitmap.Config.ARGB_8888)
        val maskCanvas = Canvas(mask)
        val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = RadialGradient(
                maskSize / 2f, maskSize / 2f,
                maskSize / 2f * 0.9f,
                intArrayOf(Color.WHITE, Color.WHITE, Color.TRANSPARENT),
                floatArrayOf(0f, 0.7f, 1f),
                Shader.TileMode.CLAMP
            )
        }
        maskCanvas.drawRect(0f, 0f, maskSize.toFloat(), maskSize.toFloat(), maskPaint)

        // Apply the mask to the swapped face
        val maskedFace = Bitmap.createBitmap(maskSize, maskSize, Bitmap.Config.ARGB_8888)
        val maskedCanvas = Canvas(maskedFace)
        maskedCanvas.drawBitmap(swappedFace, 0f, 0f, null)
        val maskApplyPaint = Paint().apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        }
        maskedCanvas.drawBitmap(mask, 0f, 0f, maskApplyPaint)

        // Draw the masked swapped face onto the result using inverse transform
        val resultCanvas = Canvas(result)
        val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        resultCanvas.drawBitmap(maskedFace, inverseMatrix, drawPaint)

        mask.recycle()
        maskedFace.recycle()
        return result
    }

    // ─── NMS (Non-Maximum Suppression) ──────────────────────────────────

    private fun nms(faces: List<DetectedFace>, threshold: Float): List<DetectedFace> {
        val sorted = faces.sortedByDescending { it.score }.toMutableList()
        val result = mutableListOf<DetectedFace>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.bbox, it.bbox) > threshold }
        }

        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        if (interLeft >= interRight || interTop >= interBottom) return 0f

        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()

        return interArea / (aArea + bArea - interArea)
    }

    // ─── ONNX Emap Extraction ───────────────────────────────────────────

    /**
     * Extract the 'emap' (buff2fs) matrix from the inswapper ONNX model.
     * This matrix is stored as an unused initializer in the ONNX protobuf.
     * The source embedding must be multiplied by this matrix before feeding
     * to the swapper model.
     *
     * Approach: scan the last portion of the ONNX file for the "buff2fs" tensor
     * name, then parse the surrounding protobuf to extract the raw float data.
     */
    private fun extractEmapFromOnnx(modelFile: File): FloatArray? {
        try {
            val raf = RandomAccessFile(modelFile, "r")
            // The emap (~1MB) is the last initializer, so read last 5MB
            val readSize = minOf(modelFile.length(), 5L * 1024 * 1024)
            val offset = modelFile.length() - readSize
            raf.seek(offset)
            val buffer = ByteArray(readSize.toInt())
            raf.readFully(buffer)
            raf.close()

            // Search for the protobuf field: name = "buff2fs"
            // In protobuf: field 8 (name), wire type 2 → tag byte = (8 << 3) | 2 = 0x42
            // Then varint length 7, then "buff2fs"
            val namePattern = byteArrayOf(
                0x42, 0x07,
                'b'.code.toByte(), 'u'.code.toByte(), 'f'.code.toByte(), 'f'.code.toByte(),
                '2'.code.toByte(), 'f'.code.toByte(), 's'.code.toByte()
            )

            val namePos = findPattern(buffer, namePattern)
            if (namePos < 0) {
                Log.w(TAG, "Could not find 'buff2fs' in ONNX model")
                return null
            }
            Log.i(TAG, "Found 'buff2fs' at position ${namePos + offset}")

            // After the name field, scan for raw_data field (field 13, wire type 2)
            // Tag = (13 << 3) | 2 = 0x6A
            var pos = namePos + namePattern.size
            while (pos < buffer.size - 10) {
                val tag = buffer[pos].toInt() and 0xFF

                if (tag == 0x6A) {
                    // Found raw_data field
                    pos++
                    val (len, bytesRead) = readVarintFromBuffer(buffer, pos)
                    pos += bytesRead

                    if (len > 0 && len <= 512 * 512 * 4 && pos + len.toInt() <= buffer.size) {
                        val numFloats = (len / 4).toInt()
                        val floats = FloatArray(numFloats)
                        val bb = ByteBuffer.wrap(buffer, pos, len.toInt())
                        bb.order(ByteOrder.LITTLE_ENDIAN)
                        bb.asFloatBuffer().get(floats)
                        Log.i(TAG, "Extracted emap raw_data: $numFloats floats")
                        return floats
                    }
                }

                // Skip this field
                val wireType = tag and 7
                pos++
                when (wireType) {
                    0 -> { // varint
                        while (pos < buffer.size && buffer[pos].toInt() and 0x80 != 0) pos++
                        pos++
                    }
                    1 -> pos += 8 // 64-bit
                    2 -> { // length-delimited
                        val (len, bytesRead) = readVarintFromBuffer(buffer, pos)
                        pos += bytesRead + len.toInt()
                    }
                    5 -> pos += 4 // 32-bit
                    else -> pos++ // unknown, skip byte
                }
            }

            Log.w(TAG, "Found buff2fs name but could not parse raw_data")
            return null
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract emap: ${e.message}", e)
            return null
        }
    }

    private fun findPattern(data: ByteArray, pattern: ByteArray): Int {
        outer@ for (i in 0..data.size - pattern.size) {
            for (j in pattern.indices) {
                if (data[i + j] != pattern[j]) continue@outer
            }
            return i
        }
        return -1
    }

    private fun readVarintFromBuffer(buffer: ByteArray, startPos: Int): Pair<Long, Int> {
        var result = 0L
        var shift = 0
        var pos = startPos
        while (pos < buffer.size) {
            val b = buffer[pos].toInt() and 0xFF
            result = result or ((b.toLong() and 0x7F) shl shift)
            pos++
            if (b and 0x80 == 0) break
            shift += 7
        }
        return Pair(result, pos - startPos)
    }
}
