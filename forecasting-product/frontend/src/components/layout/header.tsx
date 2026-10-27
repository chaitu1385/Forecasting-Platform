"use client";

import { useAuth } from "@/providers/auth-provider";
import { Moon, Sun, LogOut, User } from "lucide-react";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

export function Header() {
  const { user, isAuthenticated, mode, logout } = useAuth();
  const router = useRouter();
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    if (stored === "dark") {
      setDark(true);
      document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };

  return (
    <header className="flex h-14 items-center justify-between border-b bg-background px-6">
      {/* Left: breadcrumb area */}
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-semibold text-foreground">
          Sales Forecasting Product
        </h1>
        {mode === "demo" && (
          <span className="rounded border border-amber-500 px-2 py-1 text-xs font-medium">
            Demo — authentication disabled
          </span>
        )}
      </div>

      {/* Right: user info + controls */}
      <div className="flex items-center gap-3">
        {/* Dark mode toggle */}
        <button
          onClick={toggleTheme}
          className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title={dark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>

        {isAuthenticated && user && (
          <>
            {/* User badge */}
            <div className="flex items-center gap-2 rounded-md bg-muted px-3 py-1.5">
              <User className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-sm font-medium">{user.username}</span>
              <span className="rounded bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary">
                {user.role.replace("_", " ")}
              </span>
            </div>

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="rounded-md p-2 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
              title="Logout"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </>
        )}
      </div>
    </header>
  );
}
