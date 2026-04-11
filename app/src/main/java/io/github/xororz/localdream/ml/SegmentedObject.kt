package io.github.xororz.localdream.ml

import android.graphics.Bitmap

data class SegmentedObject(
    val id: Int,
    val label: String,
    val confidence: Float,
    val bbox: BoundingBox,
    val mask: Bitmap,
    val maskData: List<Int>
)

data class BoundingBox(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int
)

data class ImageObjects(
    val imageWidth: Int,
    val imageHeight: Int,
    val objects: List<SegmentedObject>,
    val json: String
) {
    fun toJson(): String = json
    
    companion object {
        fun fromJson(json: String, imageWidth: Int, imageHeight: Int): ImageObjects {
            return try {
                val cleanJson = json.trim()
                val objects = mutableListOf<SegmentedObject>()
                val regex = """\{[^}]+\}""".toRegex()
                val matches = regex.findAll(cleanJson)
                
                var id = 0
                matches.forEach { match ->
                    val obj = match.value
                    val labelRegex = """"label"\s*:\s*"([^"]+)"""".toRegex()
                    val scoreRegex = """"score"\s*:\s*([0-9.]+)""".toRegex()
                    val bboxRegex = """bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]""".toRegex()
                    
                    val label = labelRegex.find(obj)?.groupValues?.get(1) ?: "object"
                    val score = scoreRegex.find(obj)?.groupValues?.get(1)?.toFloatOrNull() ?: 0.9f
                    val bboxMatch = bboxRegex.find(obj)
                    
                    if (bboxMatch != null) {
                        val x = bboxMatch.groupValues[1].toInt()
                        val y = bboxMatch.groupValues[2].toInt()
                        val w = bboxMatch.groupValues[3].toInt()
                        val h = bboxMatch.groupValues[4].toInt()
                        
                        objects.add(
                            SegmentedObject(
                                id = id++,
                                label = label,
                                confidence = score,
                                bbox = BoundingBox(x, y, w, h),
                                mask = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888),
                                maskData = emptyList()
                            )
                        )
                    }
                }
                
                ImageObjects(
                    imageWidth = imageWidth,
                    imageHeight = imageHeight,
                    objects = objects,
                    json = json
                )
            } catch (e: Exception) {
                ImageObjects(imageWidth, imageHeight, emptyList(), json)
            }
        }
    }
}
