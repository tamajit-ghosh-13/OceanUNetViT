import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OceanEmbed | Realtime Neural AI & Python Backend",
  description: "Next.js frontend with high-performance Python telemetry backend",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full bg-slate-950 text-slate-100 font-sans">
        {children}
      </body>
    </html>
  );
}
