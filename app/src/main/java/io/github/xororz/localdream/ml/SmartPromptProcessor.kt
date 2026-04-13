package io.github.xororz.localdream.ml

/**
 * Smart Prompt Processor for inpainting.
 * 
 * Converts natural language edit instructions into optimized Stable Diffusion
 * inpaint prompts with appropriate denoise strength recommendations.
 * 
 * Example: "change the girl's eyes to red" →
 *   prompt: "girl with vivid red eyes, detailed iris, sharp focus"
 *   denoise: 0.45
 *   targetKeywords: ["eyes", "eye"]
 */
object SmartPromptProcessor {

    data class ProcessedPrompt(
        val inpaintPrompt: String,        // Optimized SD prompt
        val negativePrompt: String,       // Negative prompt for better results
        val denoiseStrength: Float,       // 0.0-1.0, how much to change
        val cfgScale: Float,             // Guidance scale
        val targetKeywords: List<String>, // Keywords to match objects
        val editType: EditType            // Type of edit for UI hints
    )

    enum class EditType {
        COLOR_CHANGE,     // "change X color to Y" → low denoise
        STYLE_CHANGE,     // "make X look like Y" → medium denoise
        CLOTHING_CHANGE,  // "change outfit/shirt/dress" → medium-high denoise
        REMOVAL,          // "remove X" → high denoise
        ADDITION,         // "add X to Y" → high denoise
        REPLACEMENT,      // "replace X with Y" → high denoise
        ENHANCEMENT,      // "make X more beautiful" → low denoise
        GENERAL           // fallback
    }

    // Body/person part keywords for targeting
    private val bodyParts = mapOf(
        "eye" to listOf("eyes", "eye", "iris", "pupil"),
        "hair" to listOf("hair", "hairstyle", "bangs", "locks"),
        "face" to listOf("face", "facial", "cheek", "cheeks"),
        "lip" to listOf("lips", "lip", "mouth"),
        "nose" to listOf("nose"),
        "ear" to listOf("ears", "ear"),
        "hand" to listOf("hands", "hand", "fingers", "finger"),
        "skin" to listOf("skin", "complexion"),
        "nail" to listOf("nails", "nail", "fingernails")
    )

    // Clothing keywords
    private val clothingParts = mapOf(
        "shirt" to listOf("shirt", "top", "blouse", "t-shirt", "tshirt"),
        "dress" to listOf("dress", "gown", "outfit"),
        "pants" to listOf("pants", "trousers", "jeans", "shorts"),
        "skirt" to listOf("skirt", "miniskirt"),
        "shoes" to listOf("shoes", "shoe", "boots", "boot", "heels", "sneakers"),
        "jacket" to listOf("jacket", "coat", "hoodie", "sweater"),
        "hat" to listOf("hat", "cap", "headwear"),
        "glasses" to listOf("glasses", "sunglasses", "spectacles"),
        "underwear" to listOf("underwear", "bra", "panties", "lingerie", "bikini"),
        "swimsuit" to listOf("swimsuit", "swimwear", "bikini"),
        "clothing" to listOf("clothing", "clothes", "outfit", "attire", "wardrobe")
    )

    // Color keywords
    private val colors = listOf(
        "red", "blue", "green", "yellow", "purple", "pink", "orange",
        "black", "white", "brown", "gray", "grey", "gold", "golden",
        "silver", "blonde", "brunette", "auburn", "platinum", "violet",
        "crimson", "scarlet", "azure", "emerald", "ruby", "sapphire",
        "teal", "cyan", "magenta", "lavender", "coral", "ivory"
    )

    // Action verbs that indicate edit type
    private val removeActions = listOf("remove", "delete", "erase", "get rid of", "take off", "strip")
    private val addActions = listOf("add", "put", "place", "give")
    private val changeActions = listOf("change", "make", "turn", "transform", "convert", "switch")
    private val enhanceActions = listOf("improve", "enhance", "beautify", "fix", "refine", "sharpen")
    private val replaceActions = listOf("replace", "swap", "substitute")
    private val undressActions = listOf("undress", "naked", "nude", "topless", "bare", "strip")

    /**
     * Process a natural language edit instruction into an optimized inpaint prompt.
     */
    fun process(userInstruction: String): ProcessedPrompt {
        val instruction = userInstruction.trim().lowercase()
        
        // Detect edit type
        val editType = detectEditType(instruction)
        
        // Extract target keywords
        val targetKeywords = extractTargetKeywords(instruction)
        
        // Extract color if present
        val targetColor = extractColor(instruction)
        
        // Extract the subject/object being described
        val subject = extractSubject(instruction)
        
        // Build optimized prompt
        val (prompt, negative) = buildOptimizedPrompt(
            instruction, editType, targetKeywords, targetColor, subject
        )
        
        // Determine denoise strength based on edit type
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
            editType = editType
        )
    }

    private fun detectEditType(instruction: String): EditType {
        // Check for undress/nude actions first
        if (undressActions.any { instruction.contains(it) }) {
            return EditType.CLOTHING_CHANGE
        }
        
        // Check for color changes: "change X color to Y", "make X red"
        val hasColor = colors.any { instruction.contains(it) }
        val hasColorAction = changeActions.any { instruction.contains(it) }
        if (hasColor && hasColorAction) {
            // Check if it's about clothing or body parts
            val hasClothing = clothingParts.values.flatten().any { instruction.contains(it) }
            if (hasClothing) return EditType.CLOTHING_CHANGE
            return EditType.COLOR_CHANGE
        }
        
        if (removeActions.any { instruction.contains(it) }) return EditType.REMOVAL
        if (addActions.any { instruction.contains(it) }) return EditType.ADDITION
        if (replaceActions.any { instruction.contains(it) }) return EditType.REPLACEMENT
        if (enhanceActions.any { instruction.contains(it) }) return EditType.ENHANCEMENT
        if (hasColor) return EditType.COLOR_CHANGE
        
        // Check for clothing keywords
        if (clothingParts.values.flatten().any { instruction.contains(it) }) {
            return EditType.CLOTHING_CHANGE
        }
        
        return EditType.GENERAL
    }

    private fun extractTargetKeywords(instruction: String): List<String> {
        val keywords = mutableListOf<String>()
        
        // Check body parts
        for ((_, variants) in bodyParts) {
            for (variant in variants) {
                if (instruction.contains(variant)) {
                    keywords.addAll(variants)
                    break
                }
            }
        }
        
        // Check clothing
        for ((_, variants) in clothingParts) {
            for (variant in variants) {
                if (instruction.contains(variant)) {
                    keywords.addAll(variants)
                    break
                }
            }
        }
        
        // Also add person-related keywords if referring to a person
        val personWords = listOf("girl", "boy", "man", "woman", "person", "her", "his", "she", "he")
        if (personWords.any { instruction.contains(it) }) {
            keywords.addAll(listOf("person", "figure", "body"))
        }
        
        return keywords.distinct()
    }

    private fun extractColor(instruction: String): String? {
        // Find the color mentioned, preferring later mentions (usually the target)
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
        // Try to extract what is being described
        // Look for patterns like "the girl", "her eyes", "the dress"
        val patterns = listOf(
            Regex("(?:the |a |an |her |his |their )(\\w+(?:'s \\w+)?)"),
            Regex("(?:of |on )(\\w+)")
        )
        
        for (pattern in patterns) {
            val match = pattern.find(instruction)
            if (match != null) {
                return match.groupValues[1]
            }
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
                // For color changes, describe the desired result
                val bodyPart = targetKeywords.firstOrNull() ?: subject
                if (targetColor != null) {
                    promptParts.add("$targetColor $bodyPart")
                    promptParts.add("vivid $targetColor color")
                    promptParts.add("detailed")
                    
                    // Add specific quality hints based on body part
                    if (bodyPart in listOf("eyes", "eye", "iris")) {
                        promptParts.add("detailed iris")
                        promptParts.add("sharp eyes")
                        promptParts.add("beautiful $targetColor eyes")
                    } else if (bodyPart in listOf("hair", "hairstyle")) {
                        promptParts.add("beautiful $targetColor hair")
                        promptParts.add("flowing hair")
                        promptParts.add("shiny")
                    } else if (bodyPart in listOf("lips", "lip")) {
                        promptParts.add("$targetColor lipstick")
                        promptParts.add("glossy lips")
                    }
                }
                promptParts.add("high quality")
                promptParts.add("sharp focus")
                promptParts.add("photorealistic")
            }

            EditType.CLOTHING_CHANGE -> {
                // For clothing changes
                if (undressActions.any { instruction.contains(it) }) {
                    promptParts.add("bare skin")
                    promptParts.add("exposed body")
                    promptParts.add("no clothing")
                    promptParts.add("detailed skin texture")
                    promptParts.add("anatomically correct")
                    negativeParts.add("clothing")
                    negativeParts.add("fabric")
                } else {
                    if (targetColor != null) {
                        val clothing = clothingParts.entries
                            .firstOrNull { (_, v) -> v.any { instruction.contains(it) } }
                            ?.key ?: "clothing"
                        promptParts.add("$targetColor $clothing")
                        promptParts.add("beautiful $targetColor fabric")
                    } else {
                        // Extract the desired clothing description from the instruction
                        promptParts.add(extractDesiredDescription(instruction))
                    }
                    promptParts.add("detailed fabric texture")
                    promptParts.add("high quality")
                    promptParts.add("fashion photography")
                }
            }

            EditType.REMOVAL -> {
                promptParts.add("clean background")
                promptParts.add("empty space")
                promptParts.add("seamless")
                promptParts.add("natural fill")
                // Add target to negative prompt
                targetKeywords.forEach { negativeParts.add(it) }
            }

            EditType.ADDITION -> {
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.add("detailed")
                promptParts.add("high quality")
                promptParts.add("natural lighting")
            }

            EditType.REPLACEMENT -> {
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.add("detailed")
                promptParts.add("high quality")
                promptParts.add("realistic")
            }

            EditType.ENHANCEMENT -> {
                promptParts.add("enhanced")
                promptParts.add("high quality")
                promptParts.add("sharp focus")
                promptParts.add("detailed")
                promptParts.add("4k")
                promptParts.add("professional photography")
            }

            EditType.STYLE_CHANGE -> {
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.add("artistic")
                promptParts.add("detailed")
                promptParts.add("high quality")
            }

            EditType.GENERAL -> {
                // Use the instruction as-is but enhance it
                promptParts.add(extractDesiredDescription(instruction))
                promptParts.add("high quality")
                promptParts.add("detailed")
                promptParts.add("sharp focus")
            }
        }

        val prompt = promptParts.filter { it.isNotBlank() }.joinToString(", ")
        val negative = negativeParts.distinct().joinToString(", ")

        return Pair(prompt, negative)
    }

    /**
     * Extract what the user wants the result to look like from their instruction.
     * e.g., "change her shirt to a red silk dress" → "red silk dress"
     * e.g., "make her hair longer and blonde" → "longer blonde hair"
     */
    private fun extractDesiredDescription(instruction: String): String {
        // Try common patterns
        val patterns = listOf(
            Regex("(?:to |into |with |wearing )(a |an )?(.+)$"),
            Regex("(?:make |turn )(?:it |her |him |the \\w+ )(.+)$"),
            Regex("(?:give )(?:her |him |the \\w+ )(a |an )?(.+)$")
        )

        for (pattern in patterns) {
            val match = pattern.find(instruction)
            if (match != null) {
                return match.groupValues.last().trim()
            }
        }

        // Fallback: remove action verbs and return the rest
        var result = instruction
        val allActions = removeActions + addActions + changeActions + 
                         enhanceActions + replaceActions + undressActions
        for (action in allActions) {
            result = result.replace(action, "").trim()
        }
        // Remove common filler words
        result = result.replace(Regex("\\b(the|a|an|her|his|their|to|from|of|in|on|with)\\b"), " ")
            .replace(Regex("\\s+"), " ").trim()

        return result
    }
}
