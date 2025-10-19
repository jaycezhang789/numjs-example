import type { DocGroup } from '../../types'

type DocsSectionProps = {
  groups: DocGroup[]
}

export const DocsSection = ({ groups }: DocsSectionProps) => (
  <section className="docs-section">
    <div className="section-header">
      <h2>文档入口</h2>
      <p className="section-summary">
        根据不同主题整理的 NumJS 文档与示例，便于继续深入探索。
      </p>
    </div>
    <div className="doc-grid">
      {groups.map((group) => (
        <article key={group.title} className="card doc-card">
          <h3>{group.title}</h3>
          <p className="description">{group.description}</p>
          <ul className="doc-links">
            {group.links.map((link) => (
              <li key={link.url}>
                <a href={link.url} target="_blank" rel="noreferrer">
                  {link.label}
                </a>
                {link.note && <span className="doc-note">{link.note}</span>}
              </li>
            ))}
          </ul>
        </article>
      ))}
    </div>
  </section>
)
