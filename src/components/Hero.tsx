import type { Status } from '../types'

type HeroProps = {
  status: Status
  backend: string | null
  error: string | null
}

export const Hero = ({ status, backend, error }: HeroProps) => (
  <header className="hero">
    <h1>NumJS × Vite 示例</h1>
    <p>展示 @jayce789/numjs 在浏览器中的常见矩阵操作。</p>
    <div className={`status-pill status-${status}`}>
      {status === 'loading' && '正在加载 NumJS 后端…'}
      {status === 'ready' && `当前后端：${backend}`}
      {status === 'error' && `初始化失败：${error}`}
    </div>
  </header>
)
