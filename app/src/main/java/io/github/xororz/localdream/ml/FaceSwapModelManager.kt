package io.github.xororz.localdream.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Manages downloading of face swap ONNX models from HuggingFace.
 * Downloads: SCRFD (~16MB), ArcFace (~167MB), inswapper_128 (~555MB)
 */
class FaceSwapModelManager(private val context: Context) {

    companion object {
        private const val TAG = "FaceSwapModelManager"
    }

    sealed class DownloadState {
        object Idle : DownloadState()
        data class Downloading(val modelName: String, val progress: Float, val totalProgress: Float) : DownloadState()
        object Complete : DownloadState()
        data class Error(val message: String) : DownloadState()
    }

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState

    fun areModelsReady(): Boolean = FaceSwapEngine.areModelsDownloaded(context)

    /**
     * Download all required models. Skips any that already exist.
     */
    suspend fun downloadModels(): Boolean = withContext(Dispatchers.IO) {
        try {
            val modelDir = FaceSwapEngine.getModelDir(context)
            val models = FaceSwapEngine.MODEL_URLS
            var completedModels = 0

            for ((filename, url) in models) {
                val targetFile = File(modelDir, filename)

                if (targetFile.exists() && targetFile.length() > 0) {
                    Log.i(TAG, "Skipping $filename (already downloaded: ${targetFile.length()} bytes)")
                    completedModels++
                    _downloadState.value = DownloadState.Downloading(
                        filename,
                        1f,
                        completedModels.toFloat() / models.size
                    )
                    continue
                }

                Log.i(TAG, "Downloading $filename from $url")
                val success = downloadFile(url, targetFile, filename, completedModels, models.size)
                if (!success) {
                    _downloadState.value = DownloadState.Error("Failed to download $filename")
                    return@withContext false
                }
                completedModels++
            }

            _downloadState.value = DownloadState.Complete
            Log.i(TAG, "All models downloaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Download failed: ${e.message}", e)
            _downloadState.value = DownloadState.Error(e.message ?: "Unknown error")
            false
        }
    }

    private fun downloadFile(
        url: String,
        targetFile: File,
        modelName: String,
        completedModels: Int,
        totalModels: Int
    ): Boolean {
        val client = OkHttpClient.Builder()
            .connectTimeout(60, TimeUnit.SECONDS)
            .readTimeout(600, TimeUnit.SECONDS)
            .followRedirects(true)
            .build()

        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "LocalDream/1.0")
            .build()

        val tempFile = File(targetFile.parentFile, "${targetFile.name}.tmp")

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "HTTP ${response.code} for $url")
                    return false
                }

                val body = response.body ?: return false
                val contentLength = body.contentLength()
                Log.i(TAG, "Downloading $modelName: ${contentLength / 1024 / 1024}MB")

                body.byteStream().use { input ->
                    tempFile.outputStream().use { output ->
                        val buffer = ByteArray(8192)
                        var downloaded = 0L
                        var lastUpdate = 0L

                        while (true) {
                            val read = input.read(buffer)
                            if (read == -1) break
                            output.write(buffer, 0, read)
                            downloaded += read

                            // Update progress at most every 100ms
                            val now = System.currentTimeMillis()
                            if (now - lastUpdate > 100 && contentLength > 0) {
                                lastUpdate = now
                                val fileProgress = downloaded.toFloat() / contentLength
                                val totalProgress = (completedModels + fileProgress) / totalModels
                                _downloadState.value = DownloadState.Downloading(
                                    modelName, fileProgress, totalProgress
                                )
                            }
                        }
                    }
                }

                // Rename temp file to final file
                tempFile.renameTo(targetFile)
                Log.i(TAG, "Downloaded $modelName: ${targetFile.length()} bytes")
                true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download error for $modelName: ${e.message}", e)
            tempFile.delete()
            false
        }
    }
}
