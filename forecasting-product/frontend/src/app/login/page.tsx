"use client";

import { useRef, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/providers/auth-provider";
import type { Role } from "@/lib/types";

const ROLES: { value: Role; label: string; description: string }[] = [
  { value: "admin", label: "Admin", description: "Full access to all features" },
  { value: "data_scientist", label: "Data Scientist", description: "Run pipelines, configure models" },
  { value: "planner", label: "Demand Planner", description: "View forecasts, manage overrides" },
  { value: "manager", label: "Manager", description: "Approve overrides, view audit log" },
  { value: "viewer", label: "Viewer", description: "View forecasts and metrics" },
];

export default function LoginPage() {
  const { mode, loading: modeLoading, modeError, sessionMessage, loginWithToken, loginDemo, refreshMode } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [role, setRole] = useState<Role>("data_scientist");
  const [accessToken, setAccessToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const submission = useRef(0);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const attempt = ++submission.current;
    setError("");
    if (!mode) {
      setError("Sign-in is unavailable until the API authentication mode is verified.");
      return;
    }
    if (mode === "demo" && !username.trim()) {
      setError("Username is required");
      return;
    }
    if (mode === "production" && !accessToken.trim()) {
      setError("Access token is required");
      return;
    }
    const candidate = accessToken.trim();
    setAccessToken("");
    setLoading(true);
    try {
      if (mode === "production") await loginWithToken(candidate);
      else await loginDemo(username.trim(), role);
      if (attempt === submission.current) router.replace("/");
    } catch (failure) {
      if (attempt === submission.current) {
        setError(failure instanceof Error ? failure.message : "Sign-in could not be verified.");
      }
    } finally {
      if (attempt === submission.current) setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[80vh] items-center justify-center">
      <div className="w-full max-w-md space-y-8">
        <div className="text-center">
          <h2 className="text-2xl font-bold tracking-tight">Forecasting Product</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            {mode === "demo"
              ? "Explicit demo mode: identities and roles are simulations, not production authentication."
              : "Use an access token issued by your trusted operator. Password and SSO sign-in are not configured."}
          </p>
        </div>
        {modeLoading && <p role="status">Checking API authentication...</p>}
        {modeError && (
          <div className="space-y-3 rounded-lg border p-6">
            <p role="alert" className="text-sm text-destructive">{modeError}</p>
            <button type="button" onClick={refreshMode} className="rounded-md border px-4 py-2 text-sm">
              Retry connection
            </button>
          </div>
        )}
        {!modeLoading && mode && !modeError && (
          <form onSubmit={handleSubmit} className="space-y-6 rounded-lg border p-6 shadow-sm">
            {mode === "production" ? (
              <div className="space-y-2">
                <label htmlFor="access-token" className="text-sm font-medium">Access token</label>
                <input
                  id="access-token" type="password" autoComplete="off" spellCheck={false}
                  value={accessToken} onChange={(event) => setAccessToken(event.target.value)}
                  disabled={loading} placeholder="Paste your operator-issued token"
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
                />
                <p className="text-xs text-muted-foreground">
                  Paste only the token, without a Bearer prefix. The API verifies your identity and role.
                  Credentials stay in this tab&apos;s memory and are cleared on logout, expiry or reload.
                </p>
              </div>
            ) : (
              <>
                <div className="space-y-2">
                  <label htmlFor="username" className="text-sm font-medium">Demo username</label>
                  <input
                    id="username" type="text" value={username} maxLength={128}
                    onChange={(event) => setUsername(event.target.value)}
                    disabled={loading} placeholder="Enter your username"
                    className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="role" className="text-sm font-medium">Role</label>
                  <select
                    id="role" value={role} disabled={loading}
                    onChange={(event) => {
                      const selected = ROLES.find(item => item.value === event.target.value);
                      if (selected) setRole(selected.value);
                      else setError("Select a supported demo role.");
                    }}
                    className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
                  >
                    {ROLES.map(item => (
                      <option key={item.value} value={item.value}>{item.label} — {item.description}</option>
                    ))}
                  </select>
                </div>
                <p className="text-xs text-muted-foreground">
                  Demo roles are caller-selected and provide no identity assurance. Never use production credentials or data here.
                </p>
              </>
            )}
            {(error || sessionMessage) && (
              <p role="alert" className="text-sm text-destructive">{error || sessionMessage}</p>
            )}
            <button
              type="submit" disabled={loading}
              className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
            >
              {loading ? "Signing in..." : mode === "demo" ? "Sign In to Demo" : "Sign In"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
