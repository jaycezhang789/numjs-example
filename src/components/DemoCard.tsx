import type { DemoBlock } from '../types'
import { MatrixTable } from './MatrixTable'

type DemoCardProps = {
  block: DemoBlock
}

export const DemoCard = ({ block }: DemoCardProps) => (
  <article className="card">
    <h3>{block.title}</h3>
    <p className="description">{block.description}</p>
    <div className="expression">
      调用示例：<code>{block.expression}</code>
    </div>

    <div className="matrices">
      {block.inputs.map((input) => (
        <div key={input.label} className="matrix-block">
          <span className="matrix-label">{input.label}</span>
          <MatrixTable table={input.table} />
        </div>
      ))}
    </div>

    <div className="result">
      <span className="matrix-label">输出</span>
      <div className="outputs">
        {block.outputs.map((output) => (
          <div key={output.label} className="output-block">
            <span className="matrix-label">{output.label}</span>
            {output.table ? (
              <MatrixTable table={output.table} />
            ) : (
              <div className="scalar">
                <code>{output.scalar}</code>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>

    {block.highlight && <p className="highlight">{block.highlight}</p>}
  </article>
)
