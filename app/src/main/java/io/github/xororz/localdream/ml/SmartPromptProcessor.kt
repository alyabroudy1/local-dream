package io.github.xororz.localdream.ml

/**
 * Smart Prompt Processor for inpainting.
 * 
 * Converts natural language edit instructions into optimized Stable Diffusion
 * inpaint prompts with target location estimation for auto-masking.
 * 
 * Example: "change the girl's eyes to red" →
 *   prompt: "vivid red eyes, detailed iris, sharp focus"
 *   denoise: 0.40
 *   targetRegion: EYES → estimated at (0.5, 0.32) relative position
 */
object SmartPromptProcessor {

    data class ProcessedPrompt(
        val inpaintPrompt: String,
        val negativePrompt: String,
        val denoiseStrength: Float,
        val cfgScale: Float,
        val targetKeywords: List<String>,
        val editType: EditType,
        val targetRegion: TargetRegion
    )

    enum class EditType {
        COLOR_CHANGE,
        STYLE_CHANGE,
        CLOTHING_CHANGE,
        REMOVAL,
        ADDITION,
        REPLACEMENT,
        ENHANCEMENT,
        GENERAL
    }

    /**
     * Target region with estimated relative position in image (0.0 to 1.0).
     * For portrait-oriented images with a centered person.
     */
    enum class TargetRegion(
        val relativeX: Float, val relativeY: Float,
        val label: String
    ) {
        EYES(0.5f, 0.28f, "eyes"),
        NOSE(0.5f, 0.35f, "nose"),
        MOUTH(0.5f, 0.42f, "mouth"),
        FACE(0.5f, 0.30f, "face"),
        HAIR(0.5f, 0.12f, "hair"),
        EARS(0.5f, 0.30f, "ears"),
        NECK(0.5f, 0.48f, "neck"),
        SHIRT(0.5f, 0.62f, "upper body"),
        CHEST(0.5f, 0.58f, "chest"),
        PANTS(0.5f, 0.78f, "lower body"),
        SKIRT(0.5f, 0.75f, "skirt"),
        SHOES(0.5f, 0.93f, "feet"),
        HANDS(0.35f, 0.60f, "hands"),
        FULL_BODY(0.5f, 0.50f, "full body"),
        BACKGROUND(0.1f, 0.1f, "background"),
        CENTER(0.5f, 0.50f, "center"),
        UNKNOWN(0.5f, 0.50f, "region");
    }

    // Body/person part keywords → target region
    private val bodyPartToRegion = mapOf(
        "eye" to TargetRegion.EYES, "eyes" to TargetRegion.EYES,
        "iris" to TargetRegion.EYES, "pupil" to TargetRegion.EYES,
        "nose" to TargetRegion.NOSE,
        "mouth" to TargetRegion.MOUTH, "lips" to TargetRegion.MOUTH,
        "lip" to TargetRegion.MOUTH, "teeth" to TargetRegion.MOUTH,
        "smile" to TargetRegion.MOUTH,
        "face" to TargetRegion.FACE, "facial" to TargetRegion.FACE,
        "cheek" to TargetRegion.FACE, "cheeks" to TargetRegion.FACE,
        "forehead" to TargetRegion.FACE, "chin" to TargetRegion.FACE,
        "head" to TargetRegion.FACE,
        "hair" to TargetRegion.HAIR, "hairstyle" to TargetRegion.HAIR,
        "bangs" to TargetRegion.HAIR, "locks" to TargetRegion.HAIR,
        "ear" to TargetRegion.EARS, "ears" to TargetRegion.EARS,
        "neck" to TargetRegion.NECK, "necklace" to TargetRegion.NECK,
        "hand" to TargetRegion.HANDS, "hands" to TargetRegion.HANDS,
        "finger" to TargetRegion.HANDS, "fingers" to TargetRegion.HANDS,
        "nail" to TargetRegion.HANDS, "nails" to TargetRegion.HANDS,
        "skin" to TargetRegion.FULL_BODY, "body" to TargetRegion.FULL_BODY,
        "chest" to TargetRegion.CHEST, "breast" to TargetRegion.CHEST,
        "background" to TargetRegion.BACKGROUND, "bg" to TargetRegion.BACKGROUND
    )

    // Clothing keywords → target region
    private val clothingToRegion = mapOf(
        "shirt" to TargetRegion.SHIRT, "top" to TargetRegion.SHIRT,
        "blouse" to TargetRegion.SHIRT, "t-shirt" to TargetRegion.SHIRT,
        "tshirt" to TargetRegion.SHIRT, "sweater" to TargetRegion.SHIRT,
        "jacket" to TargetRegion.SHIRT, "coat" to TargetRegion.SHIRT,
        "hoodie" to TargetRegion.SHIRT,
        "dress" to TargetRegion.FULL_BODY, "gown" to TargetRegion.FULL_BODY,
        "outfit" to TargetRegion.FULL_BODY, "clothing" to TargetRegion.FULL_BODY,
        "clothes" to TargetRegion.FULL_BODY, "attire" to TargetRegion.FULL_BODY,
        "pants" to TargetRegion.PANTS, "trousers" to TargetRegion.PANTS,
        "jeans" to TargetRegion.PANTS, "shorts" to TargetRegion.PANTS,
        "skirt" to TargetRegion.SKIRT, "miniskirt" to TargetRegion.SKIRT,
        "shoes" to TargetRegion.SHOES, "shoe" to TargetRegion.SHOES,
        "boots" to TargetRegion.SHOES, "boot" to TargetRegion.SHOES,
        "heels" to TargetRegion.SHOES, "sneakers" to TargetRegion.SHOES,
        "hat" to TargetRegion.HAIR, "cap" to TargetRegion.HAIR,
        "glasses" to TargetRegion.EYES, "sunglasses" to TargetRegion.EYES,
        "bra" to TargetRegion.CHEST, "bikini" to TargetRegion.CHEST,
        "underwear" to TargetRegion.PANTS, "panties" to TargetRegion.PANTS,
        "lingerie" to TargetRegion.FULL_BODY
    )

    private val colors = listOf(
        "red", "blue", "green", "yellow", "purple", "pink", "orange",
        "black", "white", "brown", "gray", "grey", "gold", "golden",
        "silver", "blonde", "brunette", "auburn", "platinum", "violet",
        "crimson", "scarlet", "azure", "emerald", "ruby", "sapphire",
        "teal", "cyan", "magenta", "lavender", "coral", "ivory"
    )

    private val removeActions = listOf("remove", "delete", "erase", "get rid of", "take off")
    private val addActions = listOf("add", "put", "place", "give")
    private val changeActions = listOf("change", "make", "turn", "transform", "convert", "switch")
    private val enhanceActions = listOf("improve", "enhance", "beautify", "fix", "refine", "sharpen")
    private val replaceActions = listOf("replace", "swap", "substitute")
    private val undressActions = listOf("undress", "naked", "nude", "topless", "bare", "strip")

    fun process(userInstruction: String): ProcessedPrompt {
        val instruction = userInstruction.trim().lowercase()

        val editType = detectEditType(instruction)
        val targetKeywords = extractTargetKeywords(instruction)
        val targetColor = extractColor(instruction)
        val subject = extractSubject(instruction)
        val targetRegion = detectTargetRegion(instruction)

        val (prompt, negative) = buildOptimizedPrompt(instruction, editType, targetKeywords, targetColor, subject)

        val denoise = when (editType) {
            EditType.COLOR_CHANGE -> 0.40f
            EditType.ENHANCEMENT -> 0.35f
            EditType.STYLE_CHANGE -> 0.55f
            EditType.CLOTHING_CHANGE -> 0.65f
            EditType.REMOVAL -> 0.75f
            EditType.ADDITION -> 0.70f
            EditType.REPLACEMENT -> 0.70f
            EditType.GENERAL -> 0.55f
        }

        val cfg = when (editType) {
            EditType.COLOR_CHANGE -> 7.5f
            EditType.ENHANCEMENT -> 7.0f
            EditType.STYLE_CHANGE -> 7.5f
            EditType.CLOTHING_CHANGE -> 8.0f
            EditType.REMOVAL -> 8.5f
            EditType.ADDITION -> 8.0f
            EditType.REPLACEMENT -> 8.0f
            EditType.GENERAL -> 7.5f
        }

        return ProcessedPrompt(
            inpaintPrompt = prompt,
            negativePrompt = negative,
            denoiseStrength = denoise,
            cfgScale = cfg,
            targetKeywords = targetKeywords,
            editType = editType,
            targetRegion = targetRegion
        )
    }

    /**
     * Detect WHERE in the image the edit target is, based on keyword analysis.
     * Returns a TargetRegion with estimated relative (x, y) position.
     */
    private fun detectTargetRegion(instruction: String): TargetRegion {
        // Check body parts first (more specific)
        for ((keyword, region) in bodyPartToRegion) {
            if (instruction.contains(keyword)) return region
        }
        // Check clothing
        for ((keyword, region) in clothingToRegion) {
            if (instruction.contains(keyword)) return region
        }
        // Fallback
        if (instruction.contains("background") || instruction.contains("bg")) {
            return TargetRegion.BACKGROUND
        }
        return TargetRegion.CENTER
    }

    private fun detectEditType(instruction: String): EditType {
        if (undressActions.any { instruction.contains(it) }) return EditType.CLOTHING_CHANGE

        val hasColor = colors.any { instruction.contains(it) }
        val hasColorAction = changeActions.any { instruction.contains(it) }
        if (hasColor && hasColorAction) {
            val hasClothing = clothingToRegion.keys.any { instruction.contains(it) }
            if (hasClothing) return EditType.CLOTHING_CHANGE
            return EditType.COLOR_CHANGE
        }

        if (removeActions.any { instruction.contains(it) }) return EditType.REMOVAL
        if (addActions.any { instruction.contains(it) }) return EditType.ADDITION
        if (replaceActions.any { instruction.contains(it) }) return EditType.REPLACEMENT
        if (enhanceActions.any { instruction.contains(it) }) return EditType.ENHANCEMENT
        if (hasColor) return EditType.COLOR_CHANGE

        // "change X to Y" without color → REPLACEMENT
        if (hasColorAction && instruction.contains(" to ")) return EditType.REPLACEMENT

        if (clothingToRegion.keys.any { instruction.contains(it) }) return EditType.CLOTHING_CHANGE

        return EditType.GENERAL
    }

    private fun extractTargetKeywords(instruction: String): List<String> {
        val keywords = mutableListOf<String>()
        for ((keyword, _) in bodyPartToRegion) {
            if (instruction.contains(keyword)) keywords.add(keyword)
        }
        for ((keyword, _) in clothingToRegion) {
            if (instruction.contains(keyword)) keywords.add(keyword)
        }
        return keywords.distinct()
    }

    private fun extractColor(instruction: String): String? {
        var lastColor: String? = null
        var lastIndex = -1
        for (color in colors) {
            val idx = instruction.lastIndexOf(color)
            if (idx > lastIndex) {
                lastIndex = idx
                lastColor = color
            }
        }
        return lastColor
    }

    private fun extractSubject(instruction: String): String {
        val patterns = listOf(
            Regex("(?:the |a |an |her |his |their )(\\w+(?:'s \\w+)?)"),
            Regex("(?:of |on )(\\w+)")
        )
        for (pattern in patterns) {
            val match = pattern.find(instruction)
            if (match != null) return match.groupValues[1]
        }
        return ""
    }

    private fun buildOptimizedPrompt(
        instruction: String,
        editType: EditType,
        targetKeywords: List<String>,
        targetColor: String?,
        subject: String
    ): Pair<String, String> {
        val promptParts = mutableListOf<String>()
        val negativeParts = mutableListOf(
            "low quality", "blurry", "distorted", "deformed",
            "bad anatomy", "ugly", "disfigured"
        )

        when (editType) {
            EditType.COLOR_CHANGE -> {
                val bodyPart = targetKeywords.firstOrNull() ?: subject
                if (targetColor != null) {
                    promptParts.add("$targetColor $bodyPart")
                    promptParts.add("vivid $targetColor color")
                    if (bodyPart in listOf("eyes", "eye", "iris")) {
                        promptParts.add("detailed iris"); promptParts.add("beautiful $targetColor eyes")
                    } else if (bodyPart in listOf("hair", "hairstyle")) {
                        promptParts.add("beautiful $targetColor hair"); promptParts.add("shiny")
                    } else if (bodyPart in listOf("lips", "lip")) {
                        promptParts.add("$targetColor lipstick"); promptParts.add("glossy lips")
                    }
                }
                promptParts.addAll(listOf("detailed", "high quality", "sharp focus", "photorealistic"))
            }
            EditType.CLOTHING_CHANGE -> {
                if (undressActions.any { instruction.contains(it) }) {
                    promptParts.addAll(listOf("bare skin", "exposed body", "no clothing", "detailed skin texture"))
                    negativeParts.addAll(listOf("clothing", "fabric"))
                } else {
                    val desired = extractDesiredDescription(instruction)
                    if (targetColor != null && desired.isBlank()) {
                        val clothing = clothingToRegion.keys.firstOrNull { instruction.contains(it) } ?: "clothing"
                        promptParts.add("$targetColor $clothing")
                    } else {
                        promptParts.add(desired)
                    }
                    promptParts.addAll(listOf("detailed fabric texture", "high quality", "fashion photography"))
                }
            }
            EditType.REMOVAL -> {
                promptParts.addAll(listOf("clean background", "empty space", "seamless", "natural fill"))
                targetKeywords.forEach { negativeParts.add(it) }
            }
            EditType.REPLACEMENT -> {
                val desired = extractDesiredDescription(instruction)
                promptParts.add(desired)
                promptParts.addAll(listOf("detailed", "high quality", "realistic", "sharp focus"))
            }
            EditType.ADDITION -> {
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.addAll(listOf("detailed", "high quality", "natural lighting"))
            }
            EditType.ENHANCEMENT -> {
                promptParts.addAll(listOf("enhanced", "high quality", "sharp focus", "detailed", "4k", "professional photography"))
            }
            EditType.STYLE_CHANGE -> {
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.addAll(listOf("artistic", "detailed", "high quality"))
            }
            EditType.GENERAL -> {
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.addAll(listOf("high quality", "detailed", "sharp focus"))
            }
        }

        return Pair(
            promptParts.filter { it.isNotBlank() }.joinToString(", "),
            negativeParts.distinct().joinToString(", ")
        )
    }

    private fun extractDesiredDescription(instruction: String): String {
        val patterns = listOf(
            Regex("(?:to |into |with |wearing )(a |an )?(.+)$"),
            Regex("(?:make |turn )(?:it |her |him |the \\w+ )(.+)$"),
            Regex("(?:give )(?:her |him |the \\w+ )(a |an )?(.+)$")
        )
        for (pattern in patterns) {
            val match = pattern.find(instruction)
            if (match != null) return match.groupValues.last().trim()
        }

        var result = instruction
        val allActions = removeActions + addActions + changeActions + enhanceActions + replaceActions + undressActions
        for (action in allActions) { result = result.replace(action, "").trim() }
        result = result.replace(Regex("\\b(the|a|an|her|his|their|to|from|of|in|on|with)\\b"), " ")
            .replace(Regex("\\s+"), " ").trim()
        return result
    }
}
