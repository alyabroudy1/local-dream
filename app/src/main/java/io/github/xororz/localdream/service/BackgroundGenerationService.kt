package io.github.xororz.localdream.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.app.NotificationChannel
import android.app.NotificationManager
import android.graphics.Bitmap
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.Base64
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.concurrent.TimeUnit
import io.github.xororz.localdream.R
import java.io.File
import androidx.core.graphics.createBitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import java.io.ByteArrayOutputStream

class BackgroundGenerationService : Service() {
    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private val notificationManager by lazy { getSystemService(NOTIFICATION_SERVICE) as NotificationManager }

    companion object {
        private const val CHANNEL_ID = "image_generation_channel"
        private const val NOTIFICATION_ID = 1
        const val ACTION_STOP = "stop_generation"

        private val _generationState = MutableStateFlow<GenerationState>(GenerationState.Idle)
        val generationState: StateFlow<GenerationState> = _generationState

        private val _bitmapConsumed = MutableStateFlow(false)

        private val _isServiceRunning = MutableStateFlow(false)
        val isServiceRunning: StateFlow<Boolean> = _isServiceRunning

        fun resetState() {
            _generationState.value = GenerationState.Idle
            _bitmapConsumed.value = false
        }

        fun clearCompleteState() {
            if (_generationState.value is GenerationState.Complete) {
                _generationState.value = GenerationState.Idle
            }
        }

        fun markBitmapConsumed() {
            _bitmapConsumed.value = true
        }
    }

    sealed class GenerationState {
        object Idle : GenerationState()
        data class Progress(val progress: Float, val intermediateImage: Bitmap? = null) :
            GenerationState()

        data class Complete(val bitmap: Bitmap, val seed: Long?) : GenerationState()
        data class Error(val message: String) : GenerationState()
    }

    private fun updateState(newState: GenerationState) {
        _generationState.value = newState
    }

    override fun onCreate() {
        super.onCreate()
        Log.d("GenerationService", "service created")
        _isServiceRunning.value = true
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d("GenerationService", "service execute: ${intent?.extras}")

        startForeground(NOTIFICATION_ID, createNotification(0f))

        when (intent?.action) {
            ACTION_STOP -> {
                Log.d("GenerationService", "service stopped")
                stopSelf()
                return START_NOT_STICKY
            }
        }

        val prompt = intent?.getStringExtra("prompt")
        Log.d("GenerationService", "prompt: $prompt")

        if (prompt == null) {
            Log.e("GenerationService", "empty prompt")
            stopSelf()
            return START_NOT_STICKY
        }

        val negativePrompt = intent.getStringExtra("negative_prompt") ?: ""
        val steps = intent.getIntExtra("steps", 28)
        val cfg = intent.getFloatExtra("cfg", 7f)
        val seed = if (intent.hasExtra("seed")) intent.getLongExtra("seed", 0) else null
        val width = intent.getIntExtra("width", 512)
        val height = intent.getIntExtra("height", 512)
        val denoiseStrength = intent.getFloatExtra("denoise_strength", 0.6f)
        val useOpenCL = intent.getBooleanExtra("use_opencl", false)
        val scheduler = intent.getStringExtra("scheduler") ?: "dpm"

        val image = if (intent.getBooleanExtra("has_image", false)) {
            try {
                val tmpFile = File(applicationContext.filesDir, "tmp.txt")
                if (tmpFile.exists()) {
                    tmpFile.readText()
                } else {
                    null
                }
            } catch (e: Exception) {
                Log.e("GenerationService", "Failed to read image data", e)
                null
            }
        } else {
            null
        }
        val mask = if (intent.getBooleanExtra("has_mask", false)) {
            try {
                val maskFile = File(applicationContext.filesDir, "mask.txt")
                if (maskFile.exists()) {
                    maskFile.readText()
                } else {
                    Log.w(
                        "GenerationService",
                        "has_mask is true but mask.txt not found"
                    )
                    null
                }
            } catch (e: Exception) {
                Log.e("GenerationService", "Failed to read mask data", e)
                null
            }
        } else {
            null
        }

        Log.d("GenerationService", "params: steps=$steps, cfg=$cfg, seed=$seed")

        if (_generationState.value is GenerationState.Complete) {
            updateState(GenerationState.Idle)
        }
        _bitmapConsumed.value = false

        serviceScope.launch {
            Log.d("GenerationService", "start generation")
            runGeneration(
                prompt,
                negativePrompt,
                steps,
                cfg,
                seed,
                width,
                height,
                image,
                mask,
                denoiseStrength,
                useOpenCL,
                scheduler
            )
        }

        return START_NOT_STICKY
    }

    private suspend fun runGeneration(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        cfg: Float,
        seed: Long?,
        width: Int,
        height: Int,
        image: String?,
        mask: String?,
        denoiseStrength: Float,
        useOpenCL: Boolean,
        scheduler: String
    ) = withContext(Dispatchers.IO) {
        try {
            updateState(GenerationState.Progress(0f))

            val preferences =
                applicationContext.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            val showProcess = preferences.getBoolean("show_diffusion_process", false)
            val showStride = preferences.getInt("show_diffusion_stride", 1)

            // ── Crop-and-Stitch preprocessing ──
            // When mask is present, crop to mask's bounding box so SD gets
            // maximum resolution on the target area (the "Inpaint Only Masked" technique)
            var cropInfo: CropInfo? = null
            var originalBitmap: Bitmap? = null
            var processedImage = image
            var processedMask = mask

            if (image != null && mask != null) {
                try {
                    val cropStartTime = System.currentTimeMillis()

                    // Decode mask to find bounding box
                    val maskBytes = Base64.getDecoder().decode(mask)
                    val maskBitmap = BitmapFactory.decodeByteArray(maskBytes, 0, maskBytes.size)

                    if (maskBitmap != null) {
                        // Find mask bounding box
                        var minX = maskBitmap.width; var minY = maskBitmap.height
                        var maxX = 0; var maxY = 0
                        for (y in 0 until maskBitmap.height) {
                            for (x in 0 until maskBitmap.width) {
                                val pixel = maskBitmap.getPixel(x, y)
                                // Check if pixel is not fully transparent
                                if (pixel != Color.TRANSPARENT && (pixel and 0x00FFFFFF) != 0) {
                                    if (x < minX) minX = x
                                    if (x > maxX) maxX = x
                                    if (y < minY) minY = y
                                    if (y > maxY) maxY = y
                                }
                            }
                        }

                        val maskW = maxX - minX
                        val maskH = maxY - minY
                        val imageArea = maskBitmap.width * maskBitmap.height
                        val maskArea = maskW * maskH
                        val maskRatio = maskArea.toFloat() / imageArea

                        Log.i("BgGenService", "Mask bbox: ($minX,$minY)-($maxX,$maxY) ${maskW}x${maskH}, ratio=${String.format("%.2f", maskRatio)}")

                        // Only crop if mask covers less than 70% of image (otherwise, whole-image processing is better)
                        if (maskRatio < 0.70f && maskW > 10 && maskH > 10) {
                            // Add padding (25% of bbox size, minimum 32px)
                            val padX = maxOf(32, (maskW * 0.25f).toInt())
                            val padY = maxOf(32, (maskH * 0.25f).toInt())
                            val cropLeft = (minX - padX).coerceAtLeast(0)
                            val cropTop = (minY - padY).coerceAtLeast(0)
                            val cropRight = (maxX + padX).coerceAtMost(maskBitmap.width)
                            val cropBottom = (maxY + padY).coerceAtMost(maskBitmap.height)
                            val cropW = cropRight - cropLeft
                            val cropH = cropBottom - cropTop

                            Log.i("BgGenService", "Crop region: ($cropLeft,$cropTop)-($cropRight,$cropBottom) ${cropW}x${cropH}")

                            // Decode original image
                            val imgBytes = Base64.getDecoder().decode(image)
                            val imgBitmap = BitmapFactory.decodeByteArray(imgBytes, 0, imgBytes.size)

                            if (imgBitmap != null) {
                                originalBitmap = imgBitmap
                                cropInfo = CropInfo(cropLeft, cropTop, cropW, cropH, imgBitmap.width, imgBitmap.height)

                                // Crop image and mask
                                val croppedImg = Bitmap.createBitmap(imgBitmap, cropLeft, cropTop, cropW, cropH)
                                val croppedMask = Bitmap.createBitmap(maskBitmap, cropLeft, cropTop, cropW, cropH)

                                // Resize both to generation resolution
                                val resizedImg = Bitmap.createScaledBitmap(croppedImg, width, height, true)
                                val resizedMask = Bitmap.createScaledBitmap(croppedMask, width, height, true)

                                // Re-encode image as PNG base64
                                val imgBaos = ByteArrayOutputStream()
                                resizedImg.compress(Bitmap.CompressFormat.PNG, 100, imgBaos)
                                processedImage = Base64.getEncoder().encodeToString(imgBaos.toByteArray())

                                val maskBaos = ByteArrayOutputStream()
                                resizedMask.compress(Bitmap.CompressFormat.PNG, 100, maskBaos)
                                processedMask = Base64.getEncoder().encodeToString(maskBaos.toByteArray())

                                croppedImg.recycle()
                                croppedMask.recycle()
                                resizedImg.recycle()
                                resizedMask.recycle()

                                Log.i("BgGenService", "Crop-and-stitch: cropped ${cropW}x${cropH} → ${width}x${height} in ${System.currentTimeMillis() - cropStartTime}ms")
                            }
                        } else {
                            Log.i("BgGenService", "Mask too large (${String.format("%.0f", maskRatio * 100)}%), using whole-image inpainting")
                        }
                        maskBitmap.recycle()
                    }
                } catch (e: Exception) {
                    Log.w("BgGenService", "Crop-and-stitch preprocessing failed, using whole image: ${e.message}")
                    cropInfo = null
                    originalBitmap = null
                    processedImage = image
                    processedMask = mask
                }
            }

            val jsonObject = JSONObject().apply {
                put("prompt", prompt)
                put("negative_prompt", negativePrompt)
                put("steps", steps)
                put("cfg", cfg)
                put("use_cfg", true)
                put("width", width)
                put("height", height)
                put("denoise_strength", denoiseStrength)
                put("use_opencl", useOpenCL)
                put("scheduler", scheduler)
                put("show_diffusion_process", showProcess)
                put("show_diffusion_stride", showStride)
                seed?.let { put("seed", it) }
                processedImage?.let { put("image", it) }
                processedMask?.let { put("mask", it) }
            }

            val client = OkHttpClient.Builder()
                .connectTimeout(3600, TimeUnit.SECONDS)
                .readTimeout(3600, TimeUnit.SECONDS)
                .writeTimeout(3600, TimeUnit.SECONDS)
                .callTimeout(3600, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build()

            val request = Request.Builder()
                .url("http://localhost:8081/generate")
                .post(jsonObject.toString().toRequestBody("application/json".toMediaTypeOrNull()))
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException(
                        this@BackgroundGenerationService.getString(
                            R.string.error_request_failed,
                            response.code.toString()
                        )
                    )
                }

                response.body?.let { responseBody ->
                    Log.d("BgGenService", "Reading streaming response")

                    val reader = BufferedReader(InputStreamReader(responseBody.byteStream()))
                    var messageCount = 0

                    // Read line by line for efficiency
                    while (isActive) {
                        val readLineStart = System.currentTimeMillis()
                        val line = reader.readLine() ?: break
                        val readLineTime = System.currentTimeMillis() - readLineStart

                        if (line.startsWith("data: ")) {
                            val data = line.substring(6).trim()
                            if (data == "[DONE]") break

                            val jsonParseStart = System.currentTimeMillis()
                            val message = JSONObject(data)
                            val jsonParseTime = System.currentTimeMillis() - jsonParseStart
                            messageCount++

                            when (message.optString("type")) {
                                "progress" -> {
                                    val step = message.optInt("step")
                                    val totalSteps = message.optInt("total_steps")
                                    val progress = step.toFloat() / totalSteps

                                    val b64Img = message.optString("image")
                                    var bitmap: Bitmap? = null
                                    if (b64Img.isNotEmpty()) {
                                        try {
                                            val imageBytes = Base64.getDecoder().decode(b64Img)
                                            val pixels = IntArray(width * height)
                                            for (i in 0 until width * height) {
                                                val index = i * 3
                                                if (index + 2 < imageBytes.size) {
                                                    val r = imageBytes[index].toInt() and 0xFF
                                                    val g = imageBytes[index + 1].toInt() and 0xFF
                                                    val b = imageBytes[index + 2].toInt() and 0xFF
                                                    pixels[i] =
                                                        (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                                                }
                                            }
                                            bitmap = createBitmap(width, height)
                                            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
                                        } catch (e: Exception) {
                                            Log.e(
                                                "BgGenService",
                                                "Failed to decode intermediate image",
                                                e
                                            )
                                        }
                                    }

                                    updateState(GenerationState.Progress(progress, bitmap))
                                    updateNotification(progress)
                                }

                                "complete" -> {
                                    Log.d(
                                        "BgGenService",
                                        "=== Received complete message, parsing... ==="
                                    )
                                    Log.d(
                                        "BgGenService",
                                        "readLine took: ${readLineTime}ms, line length: ${line.length}"
                                    )
                                    Log.d(
                                        "BgGenService",
                                        "JSONObject parsing took: ${jsonParseTime}ms, data length: ${data.length}"
                                    )
                                    val completeStartTime = System.currentTimeMillis()

                                    // 1. Extract fields from JSON
                                    val extractStart = System.currentTimeMillis()
                                    val base64Image = message.optString("image")
                                    val returnedSeed =
                                        message.optLong("seed", -1).takeIf { it != -1L }

                                    // Log the actual params the backend used
                                    val backendParams = message.optJSONObject("params")
                                    if (backendParams != null) {
                                        Log.i(
                                            "BgGenService",
                                            "=== Backend confirmed generation params ==="
                                        )
                                        Log.i(
                                            "BgGenService",
                                            "Prompt: ${backendParams.optString("prompt")}"
                                        )
                                        Log.i(
                                            "BgGenService",
                                            "Negative: ${backendParams.optString("negative_prompt")}"
                                        )
                                        Log.i(
                                            "BgGenService",
                                            "Steps: ${backendParams.optInt("steps")} | " +
                                            "CFG: ${backendParams.optDouble("cfg")} | " +
                                            "Seed: ${backendParams.optLong("seed")} | " +
                                            "Scheduler: ${backendParams.optString("scheduler")} | " +
                                            "Size: ${backendParams.optInt("width")}x${backendParams.optInt("height")}"
                                        )
                                    }
                                    val resultWidth = message.optInt("width", 512)
                                    val resultHeight = message.optInt("height", 512)
                                    Log.d(
                                        "BgGenService",
                                        "JSON extraction took: ${System.currentTimeMillis() - extractStart}ms, Base64 length: ${base64Image.length}"
                                    )

                                    if (base64Image.isNullOrEmpty()) {
                                        throw IOException("no image data")
                                    }

                                    // 2. Base64 decode
                                    val decodeStartTime = System.currentTimeMillis()
                                    val imageBytes = Base64.getDecoder().decode(base64Image)
                                    Log.d(
                                        "BgGenService",
                                        "Base64 decoding took: ${System.currentTimeMillis() - decodeStartTime}ms, decoded size: ${imageBytes.size} bytes"
                                    )

                                    // 3. RGB conversion + Bitmap creation
                                    val bitmapStartTime = System.currentTimeMillis()
                                    val bitmap = createBitmap(resultWidth, resultHeight)
                                    val pixels = IntArray(resultWidth * resultHeight)

                                    for (i in 0 until resultWidth * resultHeight) {
                                        val index = i * 3
                                        val r = imageBytes[index].toInt() and 0xFF
                                        val g = imageBytes[index + 1].toInt() and 0xFF
                                        val b = imageBytes[index + 2].toInt() and 0xFF
                                        pixels[i] =
                                            (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                                    }
                                    bitmap.setPixels(
                                        pixels,
                                        0,
                                        resultWidth,
                                        0,
                                        0,
                                        resultWidth,
                                        resultHeight
                                    )
                                    Log.d(
                                        "BgGenService",
                                        "RGB conversion + Bitmap creation took: ${System.currentTimeMillis() - bitmapStartTime}ms"
                                    )

                                    // 4. Crop-and-Stitch: Composite result back onto original
                                    val finalBitmap = if (cropInfo != null && originalBitmap != null) {
                                        val stitchStart = System.currentTimeMillis()
                                        val ci = cropInfo!!

                                        // Start with a copy of the original image
                                        val composite = originalBitmap!!.copy(Bitmap.Config.ARGB_8888, true)
                                        val canvas = Canvas(composite)

                                        // Scale the generated result back to crop region size
                                        val scaledResult = Bitmap.createScaledBitmap(bitmap, ci.cropW, ci.cropH, true)

                                        // Draw the result onto the original at the crop position
                                        canvas.drawBitmap(
                                            scaledResult,
                                            ci.cropX.toFloat(),
                                            ci.cropY.toFloat(),
                                            null
                                        )

                                        scaledResult.recycle()
                                        bitmap.recycle()
                                        originalBitmap!!.recycle()
                                        originalBitmap = null

                                        Log.i("BgGenService", "Crop-and-stitch composite: ${ci.cropW}x${ci.cropH} → (${ci.cropX},${ci.cropY}) on ${ci.origW}x${ci.origH} in ${System.currentTimeMillis() - stitchStart}ms")
                                        composite
                                    } else {
                                        bitmap
                                    }

                                    Log.d(
                                        "BgGenService",
                                        "=== Total processing time for complete message: ${System.currentTimeMillis() - completeStartTime}ms, size: ${resultWidth}x${resultHeight} ==="
                                    )

                                    updateState(
                                        GenerationState.Complete(
                                            finalBitmap,
                                            returnedSeed
                                        )
                                    )

                                    Log.d(
                                        "BgGenService",
                                        "Generation completed, waiting for UI to consume bitmap"
                                    )

                                    // Wait for UI to consume the bitmap with timeout
                                    val waitStartTime = System.currentTimeMillis()
                                    val timeoutMs = 5000L // 5 seconds timeout
                                    while (!_bitmapConsumed.value && isActive) {
                                        if (System.currentTimeMillis() - waitStartTime > timeoutMs) {
                                            Log.w(
                                                "BgGenService",
                                                "Timeout waiting for bitmap consumption"
                                            )
                                            break
                                        }
                                        delay(100)
                                    }

                                    Log.d(
                                        "BgGenService",
                                        "Bitmap consumed, stopping service. Wait time: ${System.currentTimeMillis() - waitStartTime}ms"
                                    )
                                    stopSelf()
                                }

                                "error" -> {
                                    val errorMsg =
                                        message.optString("message", "unknown error")
                                    Log.e(
                                        "BgGenService",
                                        "Received error message: $errorMsg"
                                    )
                                    throw IOException(errorMsg)
                                }
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("GenerationService", "generation error", e)
            updateState(
                GenerationState.Error(
                    e.message ?: this@BackgroundGenerationService.getString(R.string.unknown_error)
                )
            )
            stopSelf()
        }
    }

    private fun createNotificationChannel() {
        val name = "Image Generation"
        val descriptionText = "Background image generation"
        val importance = NotificationManager.IMPORTANCE_LOW
        val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
            description = descriptionText
        }
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(progress: Float): Notification {

        val openAppIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(this.getString(R.string.generating_notify))
            .setContentText("Progress: ${(progress * 100).toInt()}%")
            .setProgress(100, (progress * 100).toInt(), false)
            .setSmallIcon(android.R.drawable.ic_popup_sync)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(progress: Float) {
        notificationManager.notify(NOTIFICATION_ID, createNotification(progress))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onTimeout(startId: Int) {
        super.onTimeout(startId)
        Log.e("GenerationService", "Foreground service timeout")
        updateState(GenerationState.Error("Service timeout"))
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()

        if (_generationState.value is GenerationState.Error) {
            resetState()
        }

        _isServiceRunning.value = false
        Log.d("GenerationService", "service destroyed, isServiceRunning set to false")
    }
}

private data class CropInfo(
    val cropX: Int,
    val cropY: Int,
    val cropW: Int,
    val cropH: Int,
    val origW: Int,
    val origH: Int
)