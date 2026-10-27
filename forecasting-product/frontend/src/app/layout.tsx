import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { QueryProvider } from "@/providers/query-provider";
import { AuthProvider } from "@/providers/auth-provider";
import { LobProvider } from "@/providers/lob-provider";
import { AuthBoundary } from "@/components/layout/auth-boundary";

export const metadata: Metadata = {
  title: "Forecasting Product",
  description: "Multi-frequency sales forecasting product for retail S&OP",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <QueryProvider>
          <AuthProvider>
            <LobProvider>
              <AuthBoundary>
                <div className="flex h-screen overflow-hidden">
                  <Sidebar />
                  <div className="flex flex-1 flex-col overflow-hidden">
                    <Header />
                    <main className="flex-1 overflow-y-auto p-6">{children}</main>
                  </div>
                </div>
              </AuthBoundary>
            </LobProvider>
          </AuthProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
