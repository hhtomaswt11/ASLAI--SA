import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        asl: {
          bg: "#181103",
          surface: "#2b2007",
          panel: "rgba(43, 32, 7, 0.76)",
          border: "rgba(255, 194, 15, 0.16)",
          accent: "#ffc20f",
          text: "#fff7df",
          muted: "#cfb97b",
          warn: "#f97316",
          success: "#d39b00",
        },
      },
      borderRadius: {
        xl2: "24px",
      },
      boxShadow: {
        panel: "0 20px 40px rgba(0, 0, 0, 0.28)",
      },
    },
  },
  plugins: [],
};

export default config;
