"use client";

import { useState, useCallback, useEffect, useRef } from "react";

/**
 * Generic hook for async operations with loading/error/result state.
 * Eliminates repeated try/catch/finally patterns across pages.
 */
export function useAsyncOperation<T>() {
  const [result, setResult] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);

  useEffect(() => () => { generation.current += 1; }, []);

  const run = useCallback(async (fn: () => Promise<T>) => {
    const current = ++generation.current;
    setLoading(true);
    setError(null);
    try {
      const data = await fn();
      if (current !== generation.current) return null;
      setResult(data);
      return data;
    } catch (err) {
      if (current !== generation.current) return null;
      const message = err instanceof Error ? err.message : "Operation failed";
      setError(message);
      return null;
    } finally {
      if (current === generation.current) setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    generation.current += 1;
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, run, reset, setError };
}
