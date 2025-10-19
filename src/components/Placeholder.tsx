import type { Status } from '../types'

type PlaceholderProps = {
  status: Status
  error: string | null
}

export const Placeholder = ({ status, error }: PlaceholderProps) => (
  <section className="placeholder">
    {status === 'loading' && (
      <>
        <div className="spinner" />
        <p>正在准备 NumJS，请稍候…</p>
      </>
    )}

    {status === 'error' && (
      <>
        <p>初始化失败，请检查控制台日志。</p>
        <pre>{error}</pre>
      </>
    )}
  </section>
)
