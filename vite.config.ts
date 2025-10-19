import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const thisFilePath = fileURLToPath(import.meta.url)
const thisDir = path.dirname(thisFilePath)

const numjsWasmDir = path.resolve(
  thisDir,
  'node_modules/@jayce789/numjs/dist/bindings/wasm',
)

const keepFiles = new Set(['num_rs_wasm.js', 'num_rs_wasm_bg.wasm'])

const numjsWasmAssets = (): Plugin => ({
  name: 'numjs-wasm-assets',
  apply: 'build',
  async generateBundle() {
    try {
      const entries = await fs.readdir(numjsWasmDir, { withFileTypes: true })

      await Promise.all(
        entries
          .filter(
            (entry) => entry.isFile() && keepFiles.has(entry.name),
          )
          .map(async (entry) => {
            const source = await fs.readFile(
              path.join(numjsWasmDir, entry.name),
            )
            this.emitFile({
              type: 'asset',
              fileName: `assets/bindings/wasm/${entry.name}`,
              source,
            })
          }),
      )
    } catch (error) {
      this.warn(
        '[numjs-wasm-assets] 无法复制 WASM 资源，请确认依赖是否正确安装。',
      )
      if (error instanceof Error) {
        this.warn(error.message)
      }
    }
  },
})

export default defineConfig({
  plugins: [react(), numjsWasmAssets()],
  optimizeDeps: {
    exclude: ['@jayce789/numjs'],
  },
})
