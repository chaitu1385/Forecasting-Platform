"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useForecast(lob: string, params?: { series_id?: string; horizon?: number; run_id?: string; legacy?: boolean }) {
  return useQuery({
    queryKey: ["forecast", lob, params],
    queryFn: () => api.getForecasts(lob, params),
    enabled: !!lob,
  });
}

export function useForecastSeries(lob: string, seriesId: string, runId?: string) {
  return useQuery({
    queryKey: ["forecast", lob, seriesId, runId],
    queryFn: () => api.getForecastSeries(lob, seriesId, runId),
    enabled: !!lob && !!seriesId,
  });
}
