import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// why proxy /ws and /health to the FastAPI backend instead of hardcoding
// an absolute URL in the client: this lets the frontend dev server (port
// 5173) and the backend (port 8000) be treated as one origin during
// development, which avoids CORS/WebSocket-origin headaches entirely -
// see docs/11-websocket-protocol.md.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
      "/health": "http://127.0.0.1:8000",
      "/providers": "http://127.0.0.1:8000",
    },
  },
});
