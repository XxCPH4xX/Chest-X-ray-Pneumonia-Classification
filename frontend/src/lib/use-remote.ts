"use client";
import { useCallback, useEffect, useState } from "react";
import { request } from "./api";

export function useRemote<T>(path: string | null) {
  const [revision, setRevision] = useState(0);
  const [result, setResult] = useState<{
    key: string;
    data: T | null;
    error: string | null;
  }>({ key: "", data: null, error: null });
  const key = `${path}:${revision}`;
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    request<T>(path, { signal: controller.signal }).then(
      (data) => {
        if (!controller.signal.aborted) setResult({ key, data, error: null });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ key, data: null, error: error.message });
      },
    );
    return () => controller.abort();
  }, [path, key]);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  return {
    data: result.key === key ? result.data : null,
    error: result.key === key ? result.error : null,
    loading: path !== null && result.key !== key,
    refresh,
  };
}
