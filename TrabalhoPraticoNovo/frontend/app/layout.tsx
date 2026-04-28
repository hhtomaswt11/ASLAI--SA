import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ASLAI Translator",
  description:
    "American Sign Language translation prototype with webcam-based gesture recognition.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-PT">
      <body>{children}</body>
    </html>
  );
}
