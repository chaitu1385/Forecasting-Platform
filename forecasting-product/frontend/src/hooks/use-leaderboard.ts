"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useLeaderboard(lob: string, runType: string = "backtest", runId?: string) {
  return useQuery({
    queryKey: ["leaderboard", lob, runType, runId],
    queryFn: () => api.getLeaderboard(lob, runType, runId),
    enabled: !!lob,
  });
}
