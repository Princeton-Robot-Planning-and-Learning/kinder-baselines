import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  base: '/static/',
  build: {
    outDir: '../src/kinder_blockly/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/reset':      'http://localhost:5000',
      '/run':        'http://localhost:5000',
      '/challenges': 'http://localhost:5000',
      '/score':      'http://localhost:5000',
    }
  }
});
