"use client";

import {
  createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError, AUTH_INVALIDATED_EVENT, setApiSession } from "@/lib/api-client";
import { ROLE_PERMISSIONS } from "@/lib/constants";
import type { AuthIdentity, AuthStatus, Role, User } from "@/lib/types";

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  mode: AuthStatus["mode"] | null;
  loading: boolean;
  modeError: string | null;
  sessionMessage: string | null;
  loginWithToken: (token: string) => Promise<void>;
  loginDemo: (username: string, role: Role) => Promise<void>;
  refreshMode: () => void;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validStatus(value: unknown): value is AuthStatus {
  return isRecord(value) && (
    (value.mode === "production" && value.auth_required === true)
    || (value.mode === "demo" && value.auth_required === false)
  );
}

function validIdentity(value: unknown): value is AuthIdentity {
  if (!isRecord(value) || typeof value.username !== "string" || !value.username.trim()
      || typeof value.email !== "string" || typeof value.role !== "string"
      || !Object.prototype.hasOwnProperty.call(ROLE_PERMISSIONS, value.role)
      || !Array.isArray(value.permissions) || !value.permissions.every(item => typeof item === "string")) {
    return false;
  }
  return (value.mode === "demo" && value.expires_at === null)
    || (value.mode === "production" && typeof value.expires_at === "number"
      && Number.isInteger(value.expires_at) && value.expires_at * 1000 > Date.now());
}

function signInMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 401) return "The access token is invalid or expired.";
  if (error instanceof ApiError && error.status === 403) return "This identity is inactive or access is denied.";
  return "Sign-in could not be verified. Check the API connection and try again.";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<User | null>(null);
  const [mode, setMode] = useState<AuthStatus["mode"] | null>(null);
  const [loading, setLoading] = useState(true);
  const [modeError, setModeError] = useState<string | null>(null);
  const [sessionMessage, setSessionMessage] = useState<string | null>(null);
  const [expiresAt, setExpiresAt] = useState<number | null>(null);
  const generation = useRef(0);
  const modeGeneration = useRef(0);
  const pendingLogin = useRef<AbortController | null>(null);
  const pendingMode = useRef<AbortController | null>(null);

  const clearAuthentication = useCallback((message: string | null = null) => {
    generation.current += 1;
    pendingLogin.current?.abort();
    pendingLogin.current = null;
    setApiSession(null);
    setUser(null);
    setExpiresAt(null);
    setSessionMessage(message);
    void queryClient.cancelQueries();
    queryClient.clear();
  }, [queryClient]);

  const loadMode = useCallback(async () => {
    const attempt = ++modeGeneration.current;
    pendingMode.current?.abort();
    const controller = new AbortController();
    pendingMode.current = controller;
    setLoading(true);
    setMode(null);
    setModeError(null);
    try {
      const result = await api.authStatus(controller.signal);
      if (controller.signal.aborted || attempt !== modeGeneration.current) return;
      if (!validStatus(result)) throw new Error("Invalid authentication mode");
      setMode(result.mode);
    } catch {
      if (!controller.signal.aborted && attempt === modeGeneration.current) {
        setModeError("Authentication settings are unavailable. Check the API connection; sign-in is disabled.");
      }
    } finally {
      if (attempt === modeGeneration.current && !controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Legacy persisted credentials are removed without ever reading or trusting them.
    try {
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
      console.warn("Legacy authentication storage could not be removed. Clear browser site data.");
    }
    const onInvalidated = () => {
      clearAuthentication("Session expired or invalid. Sign in again.");
      void loadMode();
    };
    const onPageHide = () => clearAuthentication();
    window.addEventListener(AUTH_INVALIDATED_EVENT, onInvalidated);
    window.addEventListener("pagehide", onPageHide);
    void loadMode();
    return () => {
      generation.current += 1;
      modeGeneration.current += 1;
      pendingLogin.current?.abort();
      pendingMode.current?.abort();
      setApiSession(null);
      window.removeEventListener(AUTH_INVALIDATED_EVENT, onInvalidated);
      window.removeEventListener("pagehide", onPageHide);
    };
  }, [clearAuthentication, loadMode]);

  useEffect(() => {
    if (expiresAt === null) return;
    const timer = window.setTimeout(
      () => clearAuthentication("Session expired. Sign in again."),
      Math.max(0, expiresAt * 1000 - Date.now()),
    );
    return () => window.clearTimeout(timer);
  }, [expiresAt, clearAuthentication]);

  const loginWithToken = useCallback(async (supplied: string) => {
    clearAuthentication();
    if (mode !== "production") throw new Error("Production token sign-in is not available in this API mode.");
    const token = supplied.trim();
    if (!token) throw new Error("Access token is required");
    const attempt = generation.current;
    const controller = new AbortController();
    pendingLogin.current = controller;
    try {
      const identity = await api.getCurrentUser(token, controller.signal);
      if (controller.signal.aborted || attempt !== generation.current) throw new Error("Sign-in cancelled");
      if (!validIdentity(identity) || identity.mode !== "production" || identity.expires_at === null) {
        throw new ApiError(401, "Identity was not verified");
      }
      setApiSession({ kind: "bearer", token, expiresAt: identity.expires_at });
      setUser({ username: identity.username, role: identity.role, permissions: [...identity.permissions] });
      setExpiresAt(identity.expires_at);
    } catch (error) {
      if (controller.signal.aborted || attempt !== generation.current) throw new Error("Sign-in cancelled");
      const message = signInMessage(error);
      setSessionMessage(message);
      throw new Error(message);
    } finally {
      if (pendingLogin.current === controller) pendingLogin.current = null;
    }
  }, [clearAuthentication, mode]);

  const loginDemo = useCallback(async (supplied: string, role: Role) => {
    clearAuthentication();
    if (mode !== "demo") throw new Error("Demo sign-in is not enabled by this API.");
    const username = supplied.trim();
    if (!username) throw new Error("Username is required");
    const attempt = generation.current;
    const controller = new AbortController();
    pendingLogin.current = controller;
    try {
      const identity = await api.getDemoUser(username, role, controller.signal);
      if (controller.signal.aborted || attempt !== generation.current) throw new Error("Sign-in cancelled");
      if (!validIdentity(identity) || identity.mode !== "demo"
          || identity.username !== username || identity.role !== role) {
        throw new Error("Demo identity was not verified");
      }
      setApiSession({ kind: "demo", username: identity.username, role: identity.role });
      setUser({ username: identity.username, role: identity.role, permissions: [...identity.permissions] });
    } catch (error) {
      if (controller.signal.aborted || attempt !== generation.current) throw new Error("Sign-in cancelled");
      const message = signInMessage(error);
      setSessionMessage(message);
      throw new Error(message);
    } finally {
      if (pendingLogin.current === controller) pendingLogin.current = null;
    }
  }, [clearAuthentication, mode]);

  const refreshMode = useCallback(() => {
    clearAuthentication();
    void loadMode();
  }, [clearAuthentication, loadMode]);
  const logout = useCallback(() => clearAuthentication(), [clearAuthentication]);
  const hasPermission = useCallback(
    (permission: string) => user?.permissions.includes(permission) ?? false,
    [user],
  );

  return (
    <AuthContext.Provider value={{
      user, isAuthenticated: user !== null, mode, loading, modeError, sessionMessage,
      loginWithToken, loginDemo, refreshMode, logout, hasPermission,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("AuthProvider is required");
  return context;
}
