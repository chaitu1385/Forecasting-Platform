import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import LoginPage from "@/app/login/page";
import { AuthProvider, useAuth } from "@/providers/auth-provider";
import { api, setApiSession } from "@/lib/api-client";
import { ROLE_PERMISSIONS } from "@/lib/constants";
import type { AuthIdentity, Role } from "@/lib/types";

const navigation = vi.hoisted(() => ({ replace: vi.fn(), push: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
  usePathname: () => "/login",
  useSearchParams: () => new URLSearchParams(),
}));

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function identity(username = "verified-viewer", role: Role = "viewer", mode: "production" | "demo" = "production"): AuthIdentity {
  return {
    mode, username, email: "", role, permissions: ROLE_PERMISSIONS[role],
    expires_at: mode === "production" ? Math.floor(Date.now() / 1000) + 3600 : null,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}

let mode: "production" | "demo";
let verify: (headers: Headers) => Promise<Response>;
let forecast: (headers: Headers) => Promise<Response>;
const fetchMock = vi.fn<typeof fetch>();

function wrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}><AuthProvider>{children}</AuthProvider></QueryClientProvider>;
  };
}

function authHook() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
  return { ...renderHook(() => useAuth(), { wrapper: wrapper(client) }), client };
}

beforeEach(() => {
  mode = "production";
  verify = async () => response(identity());
  forecast = async () => response({ lob: "synthetic", series_count: 0, points: [] });
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input, options) => {
    const path = new URL(String(input)).pathname;
    if (path === "/auth/status") return response({ mode, auth_required: mode === "production" });
    if (path === "/auth/me") return verify(new Headers(options?.headers));
    if (path.startsWith("/forecast/")) return forecast(new Headers(options?.headers));
    throw new Error("Unexpected test endpoint");
  });
  vi.stubGlobal("fetch", fetchMock);
  setApiSession(null);
  localStorage.clear();
  sessionStorage.clear();
  navigation.replace.mockClear();
});

afterEach(() => {
  cleanup();
  setApiSession(null);
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("production operator-token authentication", () => {
  it("shows a password-style token input, never a production role selector", async () => {
    const client = new QueryClient();
    render(<LoginPage />, { wrapper: wrapper(client) });
    const input = await screen.findByLabelText("Access token");
    expect(input).toHaveAttribute("type", "password");
    expect(screen.queryByLabelText("Role")).not.toBeInTheDocument();
    fireEvent.change(input, { target: { value: "opaque-synthetic-credential" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign In", exact: true }));
    await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith("/"));
    expect(input).toHaveValue("");
    expect(localStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("user")).toBeNull();
    expect(sessionStorage.length).toBe(0);
  });

  it("uses only the server-verified identity and role and keeps credentials out of caches", async () => {
    const { result, client } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    verify = async headers => {
      expect(headers.get("authorization")).toBe("Bearer opaque-synthetic-credential");
      return response(identity("operator-assigned", "viewer"));
    };
    await act(async () => { await result.current.loginWithToken("opaque-synthetic-credential"); });
    expect(result.current.user?.username).toBe("operator-assigned");
    expect(result.current.user?.role).toBe("viewer");
    expect(result.current.hasPermission("view_forecasts")).toBe(true);
    expect(result.current.hasPermission("manage_users")).toBe(false);
    forecast = async headers => {
      expect(headers.get("authorization")).toBe("Bearer opaque-synthetic-credential");
      return response({ lob: "synthetic", series_count: 0, points: [] });
    };
    await api.getForecasts("synthetic");
    expect(client.getQueryCache().getAll()).toHaveLength(0);
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });

  it("removes legacy persisted authentication without restoring or trusting it", async () => {
    localStorage.setItem("access_token", "obsolete-synthetic-value");
    localStorage.setItem("user", JSON.stringify({ username: "unverified-admin", role: "admin" }));
    const { result } = authHook();
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.isAuthenticated).toBe(false);
    expect(localStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("user")).toBeNull();
    forecast = async headers => {
      expect(headers.has("authorization")).toBe(false);
      return response({ lob: "synthetic", series_count: 0, points: [] });
    };
    await api.getForecasts("synthetic");
  });

  it.each([401, 403, 500])("clears the previous identity/cache after a rejected replacement (%s)", async status => {
    const { result, client } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    await act(async () => { await result.current.loginWithToken("previous-opaque-value"); });
    client.setQueryData(["protected"], { owner: "previous-user" });
    let attempts = 0;
    verify = async () => { attempts += 1; return response({ detail: "Rejected" }, status); };
    await act(async () => {
      await expect(result.current.loginWithToken("rejected-opaque-value")).rejects.toThrow();
    });
    expect(attempts).toBe(1);
    expect(result.current.isAuthenticated).toBe(false);
    expect(client.getQueryData(["protected"])).toBeUndefined();
    forecast = async headers => {
      expect(headers.has("authorization")).toBe(false);
      return response({ lob: "synthetic", series_count: 0, points: [] });
    };
    await api.getForecasts("synthetic");
  });

  it("fails closed when mode discovery is unavailable or contradictory", async () => {
    fetchMock.mockResolvedValue(response({ mode: "demo", auth_required: true }));
    const client = new QueryClient();
    render(<LoginPage />, { wrapper: wrapper(client) });
    await screen.findByRole("alert");
    expect(screen.queryByLabelText("Role")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Access token")).not.toBeInTheDocument();
    expect(navigation.replace).not.toHaveBeenCalled();
  });

  it("does not interpret an unavailable API as an anonymous demo", async () => {
    fetchMock.mockResolvedValue(response({ detail: "Unavailable" }, 500));
    const { result } = authHook();
    await waitFor(() => expect(result.current.modeError).not.toBeNull());
    expect(result.current.mode).toBeNull();
    await expect(result.current.loginDemo("untrusted", "admin")).rejects.toThrow();
    expect(result.current.isAuthenticated).toBe(false);
  });

  it.each([
    { role: "unknown" }, { permissions: null }, { expires_at: 1 },
    { mode: "demo", expires_at: null },
  ])("rejects malformed or wrong-mode identity responses (%j)", async change => {
    const { result } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    verify = async () => response({ ...identity(), ...change });
    await act(async () => {
      await expect(result.current.loginWithToken("opaque-synthetic-value")).rejects.toThrow();
    });
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("keeps the newest verified login when an older verification completes late", async () => {
    const pending = deferred<Response>();
    const { result } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    verify = async headers => headers.get("authorization")?.endsWith("first-opaque-value")
      ? pending.promise : response(identity("new-viewer", "viewer"));
    let first!: Promise<string>;
    await act(async () => {
      first = result.current.loginWithToken("first-opaque-value").then(() => "accepted", error => error.message);
    });
    await act(async () => { await result.current.loginWithToken("second-opaque-value"); });
    await act(async () => { pending.resolve(response(identity("old-admin", "admin"))); await first; });
    expect(await first).toBe("Sign-in cancelled");
    expect(result.current.user?.username).toBe("new-viewer");
    forecast = async headers => {
      expect(headers.get("authorization")).toBe("Bearer second-opaque-value");
      return response({ lob: "synthetic", series_count: 0, points: [] });
    };
    await api.getForecasts("synthetic");
  });

  it("cannot complete a pending login after logout", async () => {
    const pending = deferred<Response>();
    const { result } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    verify = () => pending.promise;
    let login!: Promise<string>;
    await act(async () => {
      login = result.current.loginWithToken("pending-opaque-value").then(() => "accepted", error => error.message);
    });
    act(() => result.current.logout());
    await act(async () => { pending.resolve(response(identity())); await login; });
    expect(await login).toBe("Sign-in cancelled");
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("drops old protected responses without logging out a newer identity", async () => {
    const pending = deferred<Response>();
    const { result, client } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    await act(async () => { await result.current.loginWithToken("first-opaque-value"); });
    forecast = () => pending.promise;
    const old = api.getForecasts("synthetic").then(() => "accepted", error => error.message);
    client.setQueryData(["protected"], { owner: "old" });
    verify = async () => response(identity("new-identity"));
    await act(async () => { await result.current.loginWithToken("new-opaque-value"); });
    pending.resolve(response({ lob: "synthetic", series_count: 1, points: [{ private: "old" }] }));
    expect(await old).toBe("Session changed. Sign in again.");
    expect(client.getQueryData(["protected"])).toBeUndefined();
    expect(result.current.user?.username).toBe("new-identity");
  });

  it("clears identity and protected caches on expiry", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
    const { result, client } = authHook();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    verify = async () => response({ ...identity(), expires_at: Math.floor(Date.now() / 1000) + 2 });
    await act(async () => { await result.current.loginWithToken("short-lived-opaque-value"); });
    client.setQueryData(["protected"], { owner: "expired" });
    await act(async () => { await vi.advanceTimersByTimeAsync(2001); });
    expect(result.current.isAuthenticated).toBe(false);
    expect(client.getQueryData(["protected"])).toBeUndefined();
  });

  it("invalidates a current 401 once and does not retry it as a network error", async () => {
    const { result, client } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    await act(async () => { await result.current.loginWithToken("opaque-synthetic-value"); });
    client.setQueryData(["protected"], { owner: "invalidated" });
    let attempts = 0;
    forecast = async () => { attempts += 1; return response({ detail: "Invalid" }, 401); };
    await act(async () => { await expect(api.getForecasts("synthetic")).rejects.toThrow(); });
    expect(attempts).toBe(1);
    expect(result.current.isAuthenticated).toBe(false);
    expect(client.getQueryData(["protected"])).toBeUndefined();
  });

  it("clears credentials and protected data on explicit logout", async () => {
    const { result, client } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("production"));
    await act(async () => { await result.current.loginWithToken("opaque-synthetic-value"); });
    client.setQueryData(["protected"], { owner: "logged-out" });
    act(() => result.current.logout());
    expect(result.current.isAuthenticated).toBe(false);
    expect(client.getQueryCache().getAll()).toHaveLength(0);
  });
});

describe("explicit demo authentication boundary", () => {
  it("shows role selection only when the server explicitly declares demo mode", async () => {
    mode = "demo";
    const client = new QueryClient();
    render(<LoginPage />, { wrapper: wrapper(client) });
    await screen.findByLabelText("Role");
    expect(screen.queryByLabelText("Access token")).not.toBeInTheDocument();
    expect(screen.getByText(/identities and roles are simulations/)).toBeInTheDocument();
  });

  it("uses server-verified demo role headers without creating fake or persisted JWTs", async () => {
    mode = "demo";
    verify = async headers => response(identity(
      decodeURIComponent(headers.get("x-demo-user") ?? ""), "viewer", "demo",
    ));
    const { result } = authHook();
    await waitFor(() => expect(result.current.mode).toBe("demo"));
    await act(async () => { await result.current.loginDemo("Demo Viewer", "viewer"); });
    forecast = async headers => {
      expect(headers.has("authorization")).toBe(false);
      expect(headers.get("x-demo-role")).toBe("viewer");
      expect(decodeURIComponent(headers.get("x-demo-user") ?? "")).toBe("Demo Viewer");
      return response({ lob: "synthetic", series_count: 0, points: [] });
    };
    await api.getForecasts("synthetic");
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });
});
