import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // IMPORTANT: replace <repo-name> with your GitHub repository name (and optionally
  // include your username) before building/deploying. Vite's `base` controls asset
  // paths in the production build so the app works when hosted under
  // https://<your-username>.github.io/<repo-name>/
  base: '/PulmoSimHost/',
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/process': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/run_local': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/demo': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/files': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    }
  }
})


