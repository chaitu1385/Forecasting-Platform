"use client";

import { useState } from "react";
import { Send, Sparkles } from "lucide-react";
import { api } from "@/lib/api-client";
import { ConfidenceBadge } from "./confidence-badge";
import type { NLQueryResponse } from "@/lib/types";

interface Props {
  seriesId: string;
  lob: string;
  runId?: string;
  legacy?: boolean;
  suggestions?: string[];
}

export function NLQueryPanel({ seriesId, lob, runId, legacy, suggestions }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<NLQueryResponse | null>(null);
  const [error, setError] = useState("");

  const defaultSuggestions = suggestions || [
    "Why is demand volatile?",
    "Is there a trend?",
    "Any anomalies?",
    "What drives seasonality?",
  ];

  const askQuestion = async (q: string) => {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.aiExplain({ series_id: seriesId, question: q, lob, run_id: runId, legacy });
      setResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get answer");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    askQuestion(question);
  };

  return (
    <div className="space-y-4 rounded-lg border p-4">
      <div className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold">Ask About This Series</h3>
      </div>

      {/* Suggestion chips */}
      <div className="flex flex-wrap gap-2">
        {defaultSuggestions.map((s) => (
          <button
            key={s}
            onClick={() => { setQuestion(s); askQuestion(s); }}
            className="rounded-full border px-3 py-1 text-xs hover:bg-muted transition-colors"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about this forecast..."
          className="flex-1 rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="rounded-md bg-primary px-3 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          Thinking...
        </div>
      )}

      {/* Error */}
      {error && <p className="text-sm text-destructive">{error}</p>}

      {/* Response */}
      {response && !loading && (
        <div className="space-y-2 rounded-md bg-muted/50 p-4">
          <div className="flex items-center gap-2">
            <ConfidenceBadge confidence={response.confidence} />
            {response.sources_used.length > 0 && (
              <span className="text-xs text-muted-foreground">
                Sources: {response.sources_used.join(", ")}
              </span>
            )}
          </div>
          <p className="text-sm leading-relaxed">{response.answer}</p>
        </div>
      )}
    </div>
  );
}
