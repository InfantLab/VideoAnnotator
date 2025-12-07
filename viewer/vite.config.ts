import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "127.0.0.1", // Force IPv4 loopback
    port: 19011,
    proxy: {
      '/api': {
        target: 'http://localhost:18011',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: 'http://localhost:18011',
        changeOrigin: true,
        secure: false,
      },
      '/docs': {
        target: 'http://localhost:18011',
        changeOrigin: true,
        secure: false,
      },
      '/openapi.json': {
        target: 'http://localhost:18011',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
