import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import fs from "node:fs";
import fsp from "node:fs/promises";

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
    {
      name: 'copy-demo-assets',
      apply: 'build',
      async closeBundle() {
        const sourceDir = path.resolve(__dirname, 'demo');
        const targetDir = path.resolve(__dirname, 'dist', 'demo');

        try {
          if (!fs.existsSync(sourceDir)) {
            return;
          }

          // Node 18+ supports fs.promises.cp
          // Copy the entire demo directory so runtime fetch('/demo/...') works in production.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const cp = (fsp as any).cp as undefined | ((src: string, dest: string, opts: any) => Promise<void>);
          if (cp) {
            await cp(sourceDir, targetDir, { recursive: true });
            return;
          }

          // Fallback: minimal recursive copy.
          const copyDir = async (src: string, dest: string) => {
            await fsp.mkdir(dest, { recursive: true });
            const entries = await fsp.readdir(src, { withFileTypes: true });
            for (const entry of entries) {
              const srcPath = path.join(src, entry.name);
              const destPath = path.join(dest, entry.name);
              if (entry.isDirectory()) {
                await copyDir(srcPath, destPath);
              } else if (entry.isFile()) {
                await fsp.copyFile(srcPath, destPath);
              }
            }
          };

          await copyDir(sourceDir, targetDir);
        } catch (error) {
          // Non-fatal: demo installs will fail gracefully at runtime if assets are missing.
          console.warn('[copy-demo-assets] Failed to copy demo assets:', error);
        }
      }
    }
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
