import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
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


