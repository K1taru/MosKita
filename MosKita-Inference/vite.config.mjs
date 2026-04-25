import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(dirname, '..');
const sharedModelsDir = path.join(repoRoot, 'models');

function getContentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === '.onnx' || ext === '.pt' || ext === '.tflite') {
    return 'application/octet-stream';
  }
  if (ext === '.json') {
    return 'application/json; charset=utf-8';
  }
  if (ext === '.yaml' || ext === '.yml') {
    return 'application/yaml; charset=utf-8';
  }

  return 'application/octet-stream';
}

function createSharedModelsMiddleware() {
  return function serveSharedModels(req, res, next) {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      next();
      return;
    }

    const requestUrl = new URL(req.url ?? '/', 'http://localhost');
    const requestPath = decodeURIComponent(requestUrl.pathname);
    const relativePath = requestPath.startsWith('/models/')
      ? requestPath.slice('/models/'.length)
      : requestPath.replace(/^\/+/, '');

    if (!relativePath) {
      next();
      return;
    }

    const filePath = path.resolve(sharedModelsDir, relativePath);
    const modelsRoot = `${sharedModelsDir}${path.sep}`;
    if (filePath !== sharedModelsDir && !filePath.startsWith(modelsRoot)) {
      res.statusCode = 403;
      res.end('Forbidden');
      return;
    }

    fs.stat(filePath, (statError, stat) => {
      if (statError || !stat.isFile()) {
        next();
        return;
      }

      res.statusCode = 200;
      res.setHeader('Content-Type', getContentType(filePath));
      res.setHeader('Content-Length', stat.size);
      res.setHeader('Cache-Control', 'no-cache');

      if (req.method === 'HEAD') {
        res.end();
        return;
      }

      fs.createReadStream(filePath).pipe(res);
    });
  };
}

function sharedModelsPlugin() {
  return {
    name: 'moskita-shared-models',
    configureServer(server) {
      server.middlewares.use('/models', createSharedModelsMiddleware());
    },
    configurePreviewServer(server) {
      server.middlewares.use('/models', createSharedModelsMiddleware());
    },
  };
}

export default defineConfig({
  plugins: [sharedModelsPlugin(), react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    fs: {
      allow: [repoRoot],
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 4173,
  },
  optimizeDeps: {
    include: ['onnxruntime-web'],
  },
  define: {
    global: 'globalThis',
  },
});
