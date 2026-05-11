import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        asl: {
          bg: "var(--asl-bg)",
          surface: "var(--asl-surface)",
          panel: "var(--asl-panel)",
          border: "var(--asl-border)",
          accent: "var(--asl-accent)",
          text: "var(--asl-text)",
          muted: "var(--asl-muted)",
          warn: "var(--asl-warn)",
          success: "var(--asl-success)",
        },
      },
      borderRadius: {
        xl2: "24px",
      },
      boxShadow: {
        panel: "0 20px 40px rgba(0, 0, 0, 0.15)",
      },
    },
  },
  plugins: [],
};

export default config;
