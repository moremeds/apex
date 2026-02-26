import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "buffer/": "buffer/",
      buffer: "buffer/",
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ["plotly.js", "react-plotly.js"],
          charts: ["lightweight-charts"],
          vendor: ["react", "react-dom", "react-router-dom"],
          tanstack: ["@tanstack/react-query", "@tanstack/react-table"],
        },
      },
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8080",
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
      },
    },
  },
})
