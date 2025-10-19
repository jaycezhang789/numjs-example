import type { DemoSection } from '../../types'
import { DemoCard } from '../DemoCard'

type SectionProps = {
  section: DemoSection
}

export const Section = ({ section }: SectionProps) => (
  <section className="demo-section">
    <div className="section-header">
      <h2>{section.heading}</h2>
      <p className="section-summary">{section.summary}</p>
    </div>
    <div className="grid">
      {section.demos.map((demo) => (
        <DemoCard key={demo.title} block={demo} />
      ))}
    </div>
  </section>
)
