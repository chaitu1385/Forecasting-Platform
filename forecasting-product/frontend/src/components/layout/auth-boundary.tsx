"use client";

import { useEffect, type ReactNode } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/providers/auth-provider";

export function AuthBoundary({ children }: { children: ReactNode }) {
  const { isAuthenticated, loading } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  useEffect(() => {
    if (!loading && !isAuthenticated && pathname !== "/login") router.replace("/login");
  }, [isAuthenticated, loading, pathname, router]);
  if (pathname === "/login") return children;
  if (loading) return <p role="status" className="p-6">Checking authentication...</p>;
  if (!isAuthenticated) return null;
  return children;
}
