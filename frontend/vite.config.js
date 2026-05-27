import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    // API는 FastAPI(8000)로 전달 — 화면은 5173에서 봄
    proxy: {
      '/upload': 'http://127.0.0.1:8000',
      '/query': 'http://127.0.0.1:8000',
      '/analyze': 'http://127.0.0.1:8000',
      '/distribution': 'http://127.0.0.1:8000',
    },
  },
})
