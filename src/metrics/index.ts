import { updateBasicMetrics } from './basicMetrics';
import { updateAdvancedMetrics } from './advancedMetrics';
import { updateCategoryMetrics } from './categoryMetrics';

export async function updateMetrics(): Promise<void> {
    // Run all updates concurrently.
    await Promise.all([/*updateBasicMetrics(), updateAdvancedMetrics(), */updateCategoryMetrics()]);
}