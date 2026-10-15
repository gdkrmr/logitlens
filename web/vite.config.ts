import { defineConfig, type Plugin } from 'vite';
import { createReadStream, statSync } from 'node:fs';
import { resolve, join, normalize } from 'node:path';

const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// dev only: LOCAL_MODELS=/path/to/dir serves GGUF files at /local-models/<relative path>
function localModels(): Plugin {
  const root = process.env.LOCAL_MODELS;
  return {
    name: 'local-models',
    configureServer(server) {
      if (!root) return;
      server.middlewares.use('/local-models', (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? '/').split('?')[0]);
        const file = normalize(join(root, rel));
        if (!file.startsWith(normalize(root)) || !file.endsWith('.gguf')) return next();
        let size: number;
        try {
          size = statSync(file).size;
        } catch {
          return next();
        }
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Accept-Ranges', 'bytes');
        res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
        const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
        if (range) {
          const start = Number(range[1]);
          const end = range[2] ? Number(range[2]) : size - 1;
          res.statusCode = 206;
          res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
          res.setHeader('Content-Length', String(end - start + 1));
          createReadStream(file, { start, end }).pipe(res);
        } else {
          res.setHeader('Content-Length', String(size));
          if (req.method === 'HEAD') return res.end();
          createReadStream(file).pipe(res);
        }
      });
    },
  };
}

export default defineConfig({
  // relative asset paths, so the build works at any URL path
  base: './',
  plugins: [localModels()],
  resolve: {
    alias: { '@wllama/wllama': resolve(import.meta.dirname, '../third_party/wllama/src/index.ts') },
  },
  server: { headers: isolationHeaders, fs: { allow: ['..'] } },
  preview: { headers: isolationHeaders },
  build: { target: 'es2022' },
});
