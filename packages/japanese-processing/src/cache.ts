/**
 * Generic LRU (Least Recently Used) cache.
 * Uses a Map which maintains insertion order for O(1) eviction.
 */

export class LRUCache<K, V> {
  private readonly map = new Map<K, V>();

  constructor(private readonly maxSize: number) {}

  get(key: K): V | undefined {
    const val = this.map.get(key);
    if (val !== undefined) {
      // Promote to most-recent position
      this.map.delete(key);
      this.map.set(key, val);
    }
    return val;
  }

  set(key: K, val: V): void {
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.maxSize) {
      // Evict oldest (first) entry
      this.map.delete(this.map.keys().next().value!);
    }
    this.map.set(key, val);
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  delete(key: K): boolean {
    return this.map.delete(key);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }

  entries(): IterableIterator<[K, V]> {
    return this.map.entries();
  }
}

// ── Shared caches (process-lifetime) ──────────────────────────────────────────

/** Cache for kuromoji token results: normalized text → token array */
export const tokenCache  = new LRUCache<string, readonly object[]>(2000);

/** Cache for dictionary lookups: word → entry */
export const dictCache   = new LRUCache<string, object | null>(3000);

/** Cache for full line analyses: text → AnalyzedLine */
export const lineCache   = new LRUCache<string, object>(1000);
