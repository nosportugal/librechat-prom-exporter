import client from 'prom-client';
import { GoogleGenAI } from '@google/genai';
import { Conversation } from '../models';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

// Cache file path for persisting category classifications
// The model name is appended as a suffix to separate caches per model
function getCacheFilePath(): string {
    const basePath = process.env.CATEGORY_CACHE_FILE || './data/category-cache.json';
    const model = process.env.VERTEX_AI_MODEL || 'gemini-1.5-flash';
    const ext = path.extname(basePath);
    const baseWithoutExt = basePath.slice(0, -ext.length);
    return `${baseWithoutExt}-${model}${ext}`;
}

// Conversation categories based on the business taxonomy
export const CONVERSATION_CATEGORIES = {
    'Uncategorized': [
        'Casual & Non-Work',
    ],
    'Technical & Engineering': [
        'Code Generation & Scaffolding',
        'Debugging & Troubleshooting',
        'Code/Query Refactoring & Optimization',
        'Technical Documentation',
        'Architecture & System Design',
        'Data Analysis & Querying',
    ],
    'Content & Communications': [
        'First Draft Creation',
        'Editing & Refinement',
        'Summarization & Extraction',
        'Translation & Localization',
    ],
    'Strategy & Analysis': [
        'Brainstorming & Ideation',
        'Market & Competitor Research',
        'Planning & Frameworks',
        'Data Interpretation & Reporting',
    ],
    'Business & Operations': [
        'HR & People Ops',
        'Legal & Compliance',
        'Sales & Customer Support',
        'Finance & Procurement',
        'Project Management',
    ],
    'Risk & Governance': [
        'Sensitive Data Query (High-Alert)',
        'Inappropriate Content Generation',
        'Quality Complaint & Correction',
    ],
    'Meta & Tool Usage': [
        'Assistant Interaction & Prompting',
    ],
} as const;

// Flat list of all valid categories for validation
export const ALL_CATEGORIES = Object.entries(CONVERSATION_CATEGORIES).flatMap(
    ([primary, subTypes]) => subTypes.map(sub => ({ primary, subType: sub })),
);

// Create the category prompt for LLM classification
function buildCategoryPrompt(title: string): string {
    // Dynamically build the category list from CONVERSATION_CATEGORIES
    const categoryList = ALL_CATEGORIES
        .map((cat, idx) => `${idx + 1}. ${cat.primary}|${cat.subType}`)
        .join('\n');

    const defaultCategory = `${ALL_CATEGORIES[0].primary}|${ALL_CATEGORIES[0].subType}`;

    return `Classify this conversation title into exactly one category.

Title: ${title}

Valid categories (copy exactly, including the | separator):
${categoryList}

Respond with ONLY the category (e.g., "${defaultCategory}").
If unsure, use: Uncategorized|Casual & Non-Work

Category:`;
}

// Gauges for category metrics
export const categoryGauges = {
    // Primary category counts
    conversationCategoryCount: new client.Gauge({
        name: 'librechat_conversation_category_count',
        help: 'Count of conversations by primary category and sub-type',
        labelNames: ['primary_category', 'sub_type'],
    }),

    // Primary category only aggregation
    conversationPrimaryCategoryCount: new client.Gauge({
        name: 'librechat_conversation_primary_category_count',
        help: 'Count of conversations by primary category',
        labelNames: ['primary_category'],
    }),

    // Category classification errors
    categoryClassificationErrors: new client.Gauge({
        name: 'librechat_category_classification_errors',
        help: 'Count of conversations that failed to be classified',
    }),

    // Last classification run timestamp
    categoryClassificationLastRun: new client.Gauge({
        name: 'librechat_category_classification_last_run_timestamp',
        help: 'Unix timestamp of the last category classification run',
    }),

    // Total conversations classified
    totalConversationsClassified: new client.Gauge({
        name: 'librechat_total_conversations_classified',
        help: 'Total number of conversations that have been classified',
    }),
};

// In-memory cache for categorized conversations to avoid re-processing
const categoryCache: Map<string, { primary: string; subType: string; title: string }> = new Map();

/**
 * Load category cache from file on startup.
 * This allows classifications to persist across service restarts.
 */
export function loadCategoryCache(): void {
    try {
        const cacheFilePath = path.resolve(getCacheFilePath());
        if (fs.existsSync(cacheFilePath)) {
            const data = fs.readFileSync(cacheFilePath, 'utf-8');
            const parsed = JSON.parse(data) as Record<string, { primary: string; subType: string; title: string }>;

            for (const [key, value] of Object.entries(parsed)) {
                categoryCache.set(key, value);
            }

            console.log(`Loaded ${categoryCache.size} cached categories from file: ${cacheFilePath}`);
        } else {
            console.log(`No existing cache file found at: ${cacheFilePath}`);
        }
    } catch (error) {
        console.error('Error loading category cache from file:', error);
    }
}

/**
 * Save category cache to file for persistence.
 * Called after new classifications are made.
 */
export function saveCategoryCache(): void {
    try {
        const cacheFilePath = path.resolve(getCacheFilePath());
        const dir = path.dirname(cacheFilePath);

        // Create directory if it doesn't exist
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        const data: Record<string, { primary: string; subType: string; title: string }> = {};
        for (const [key, value] of categoryCache.entries()) {
            data[key] = value;
        }

        fs.writeFileSync(cacheFilePath, JSON.stringify(data, null, 2));
        console.log(`Saved ${categoryCache.size} categories to cache file: ${cacheFilePath}`);
    } catch (error) {
        console.error('Error saving category cache to file:', error);
    }
}

// Load cache on module initialization
loadCategoryCache();

// Initialize GoogleGenAI client
let genAI: GoogleGenAI | null = null;

/**
 * Get or create the GoogleGenAI client.
 * For Vertex AI: Uses GOOGLE_APPLICATION_CREDENTIALS for authentication.
 * Supports both Gemini Developer API (with API key) and Vertex AI.
 */
function getGenAIClient(): GoogleGenAI {
    if (!genAI) {
        const projectId = process.env.VERTEX_AI_PROJECT_ID;
        const location = process.env.VERTEX_AI_LOCATION || 'us-central1';

        if (projectId) {
            console.log(`Using Vertex AI with project: ${projectId}, location: ${location}`);
            genAI = new GoogleGenAI({
                vertexai: true,
                project: projectId,
                location: location,
            });
        } else {
            throw new Error(
                'VERTEX_AI_PROJECT_ID environment variable is required',
            );
        }
    }
    return genAI;
}

/**
 * Classify a conversation title using Google GenAI (Gemini)
 */
async function classifyConversationTitle(title: string): Promise<{ primary: string; subType: string } | null> {
    try {
        const ai = getGenAIClient();
        const model = process.env.VERTEX_AI_MODEL || 'gemini-2.5-flash';
        const prompt = buildCategoryPrompt(title);

        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                maxOutputTokens: 512,
                temperature: 0.0, // Zero temperature for deterministic classification
                stopSequences: ['\n'], // Stop at newline to get clean output
            },
        });

        const rawText = response.text?.trim();
        //console.log(`Title: "${title}" => "${rawText}"`);

        if (!rawText) {
            console.warn(`Empty response from LLM for title: ${title}`);
            return null;
        }

        // Clean up the response: remove quotes and extra whitespace
        const text = rawText.replace(/^["'\s]+|["'\s]+$/g, '').trim();

        // Parse the response - look for the pipe separator
        const pipeIndex = text.indexOf('|');
        if (pipeIndex === -1) {
            console.warn(`Invalid category format from LLM (no pipe separator): ${rawText}`);
            return null;
        }

        const primary = text.substring(0, pipeIndex).trim();
        const subType = text.substring(pipeIndex + 1).trim();

        if (!primary || !subType) {
            console.warn(`Invalid category format from LLM (empty parts): ${rawText}`);
            return null;
        }

        // Validate the category exists
        const isValid = ALL_CATEGORIES.some(
            c => c.primary === primary && c.subType === subType,
        );

        if (!isValid) {
            console.warn(`Invalid category returned by LLM: ${primary}|${subType}`);
            return null;
        }

        return { primary, subType };
    } catch (error) {
        console.error(`Error classifying title "${title}":`, error);
        return null;
    }
}

/**
 * Batch classify conversations with rate limiting
 */
async function classifyConversationsBatch(
    conversations: Array<{ _id: string; title: string }>,
    batchSize: number = 10,
    delayMs: number = 100,
): Promise<Map<string, { primary: string; subType: string; title: string }>> {
    const results: Map<string, { primary: string; subType: string; title: string }> = new Map();

    for (let i = 0; i < conversations.length; i += batchSize) {
        const batch = conversations.slice(i, i + batchSize);

        const batchPromises = batch.map(async (conv) => {
            // Check cache first
            if (categoryCache.has(conv._id)) {
                return { id: conv._id, category: categoryCache.get(conv._id)! };
            }

            const category = await classifyConversationTitle(conv.title);
            if (category) {
                const cacheEntry = { ...category, title: conv.title };
                categoryCache.set(conv._id, cacheEntry);
                return { id: conv._id, category: cacheEntry };
            }
            else {
                console.warn(`Failed to classify conversation ID: ${conv._id}`);
                return { id: conv._id, category: null };
            }
        });

        const batchResults = await Promise.all(batchPromises);

        for (const result of batchResults) {
            if (result.category) {
                results.set(result.id, result.category);
            }
        }

        // Rate limiting delay between batches
        if (i + batchSize < conversations.length) {
            await new Promise(resolve => setTimeout(resolve, delayMs));
        }

        console.log(`Classified batch ${i / batchSize + 1} / ${Math.ceil(conversations.length / batchSize)}`);
    }

    return results;
}

/**
 * Load conversations from a local CSV file.
 * Expected CSV format: _id,title (with header row)
 */
async function loadConversationsFromCSV(csvPath: string): Promise<Array<{ _id: string; title: string }>> {
    const conversations: Array<{ _id: string; title: string }> = [];
    const resolvedPath = path.resolve(csvPath);

    if (!fs.existsSync(resolvedPath)) {
        throw new Error(`CSV file not found: ${resolvedPath}`);
    }

    const fileStream = fs.createReadStream(resolvedPath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    let isFirstLine = true;
    let idIndex = 0;
    let titleIndex = 1;

    for await (const line of rl) {
        if (!line.trim()) continue;

        // Parse CSV line (handle quoted fields with commas)
        const fields: string[] = [];
        let currentField = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                if (inQuotes && line[i + 1] === '"') {
                    // Escaped quote
                    currentField += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                fields.push(currentField.trim());
                currentField = '';
            } else {
                currentField += char;
            }
        }
        fields.push(currentField.trim());

        if (isFirstLine) {
            // Parse header to find column indices
            const lowerFields = fields.map(f => f.toLowerCase().replace(/^["']|["']$/g, ''));
            const foundIdIndex = lowerFields.findIndex(f => f === '_id' || f === 'id');
            const foundTitleIndex = lowerFields.findIndex(f => f === 'title');

            if (foundIdIndex !== -1) idIndex = foundIdIndex;
            if (foundTitleIndex !== -1) titleIndex = foundTitleIndex;

            isFirstLine = false;
            continue;
        }

        const _id = fields[idIndex]?.replace(/^["']|["']$/g, '');
        const title = fields[titleIndex]?.replace(/^["']|["']$/g, '');

        if (_id && title) {
            conversations.push({ _id, title });
        }
    }

    return conversations;
}

/**
 * Update conversation category metrics
 */
export async function updateCategoryMetrics(): Promise<void> {
    try {
        // Check if GenAI is configured (either API key or Vertex AI project)
        const projectId = process.env.VERTEX_AI_PROJECT_ID;

        // Load conversations from CSV file or MongoDB
        const localCsvPath = process.env.LOCAL_CONVERSATIONS_CSV;
        let conversations: Array<{ _id: string; title: string }>;

        if (localCsvPath) {
            // Load from local CSV file
            console.log(`Loading conversations from CSV file: ${localCsvPath}`);
            conversations = await loadConversationsFromCSV(localCsvPath);
            console.log(`Loaded ${conversations.length} conversations from CSV`);
        } else {
            // TODO: paginate if there are many conversations and use updatedAt to limit to recent ones
            // Fetch conversations with titles from MongoDB
            const mongoConversations = await Conversation.find(
                { title: { $exists: true, $nin: [null, ''] } },
                { _id: 1, title: 1 },
            ).lean() as Array<{ _id: { toString(): string }; title: string }>;

            conversations = mongoConversations.map(c => ({
                _id: c._id.toString(),
                title: c.title,
            }));
        }

        if (conversations.length === 0) {
            console.log('No conversations found for category classification');
            return;
        }

        // Filter to only process new conversations (not in cache)
        const uncachedConversations = conversations.filter(
            conv => !categoryCache.has(conv._id),
        );

        console.log(`Classifying ${uncachedConversations.length} new conversations (${categoryCache.size} cached)`);

        // Classify uncached conversations
        let classificationErrors = 0;
        if (uncachedConversations.length > 0) {
            const batchSize = parseInt(process.env.CATEGORY_BATCH_SIZE || '10');
            const delayMs = parseInt(process.env.CATEGORY_BATCH_DELAY_MS || '100');

            const newClassifications = await classifyConversationsBatch(
                uncachedConversations,
                batchSize,
                delayMs,
            );

            // Count errors
            classificationErrors = uncachedConversations.length - newClassifications.size;

            // Save cache to file after new classifications
            if (newClassifications.size > 0) {
                saveCategoryCache();
            }
        }

        // Aggregate counts from cache
        const categoryCounts: Map<string, number> = new Map();
        const primaryCounts: Map<string, number> = new Map();

        for (const conv of conversations) {
            const category = categoryCache.get(conv._id);
            if (category) {
                const key = `${category.primary}|${category.subType}`;
                categoryCounts.set(key, (categoryCounts.get(key) || 0) + 1);
                primaryCounts.set(category.primary, (primaryCounts.get(category.primary) || 0) + 1);
            }
        }

        // Update metrics
        categoryGauges.conversationCategoryCount.reset();
        for (const [key, count] of categoryCounts.entries()) {
            const [primary, subType] = key.split('|');
            categoryGauges.conversationCategoryCount.set(
                { primary_category: primary, sub_type: subType },
                count,
            );
        }

        categoryGauges.conversationPrimaryCategoryCount.reset();
        for (const [primary, count] of primaryCounts.entries()) {
            categoryGauges.conversationPrimaryCategoryCount.set(
                { primary_category: primary },
                count,
            );
        }

        categoryGauges.categoryClassificationErrors.set(classificationErrors);
        categoryGauges.categoryClassificationLastRun.set(Date.now() / 1000);
        categoryGauges.totalConversationsClassified.set(categoryCache.size);

        console.log(`Category metrics updated: ${categoryCache.size} classified, ${classificationErrors} errors`);
    } catch (error) {
        console.error('Error updating category metrics:', error);
    }
}

/**
 * Clear the category cache (useful for testing or forced re-classification)
 */
export function clearCategoryCache(): void {
    categoryCache.clear();

    // Also delete the cache file
    try {
        const cacheFilePath = path.resolve(getCacheFilePath());
        if (fs.existsSync(cacheFilePath)) {
            fs.unlinkSync(cacheFilePath);
            console.log(`Deleted cache file: ${cacheFilePath}`);
        }
    } catch (error) {
        console.error('Error deleting cache file:', error);
    }

    console.log('Category cache cleared');
}
