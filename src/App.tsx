import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { documentationGroups } from './data/documentation'
import { useNumJSStatus } from './hooks/useNumJSStatus'
import { useLinearAsyncData } from './hooks/useLinearAsyncData'
import { useDemoSections } from './hooks/useDemoSections'
import type { DemoSection } from './types'
import { Section } from './components/sections/Section'
import { Hero } from './components/Hero'
import { TabBar, type TabDefinition } from './components/TabBar'
import { DocsSection } from './components/DocsSection'
import { Placeholder } from './components/Placeholder'
import { Footer } from './components/Footer'

const DOCS_TAB: TabDefinition = {
  id: 'docs',
  label: '文档资源',
  type: 'docs',
}

const ACTIVE_TAB_STORAGE_KEY = 'numjs-demo.activeTab'

const createSectionTabs = (sections: DemoSection[]): TabDefinition[] =>
  sections.map((section) => ({
    id: section.id,
    label: section.heading,
    type: 'section' as const,
  }))

function App() {
  const { status, backend, error } = useNumJSStatus()
  const linearAsyncData = useLinearAsyncData(status)
  const sections = useDemoSections(status, linearAsyncData)
  const [activeTab, setActiveTab] = useState<string | null>(null)

  const tabs = useMemo(() => {
    const sectionTabs = createSectionTabs(sections)
    return [...sectionTabs, DOCS_TAB]
  }, [sections])

  useEffect(() => {
    if (status !== 'ready' || tabs.length === 0) {
      setActiveTab(null)
      return
    }

    setActiveTab((current) => {
      if (current && tabs.some((tab) => tab.id === current)) {
        return current
      }

      if (typeof window !== 'undefined') {
        const stored = window.localStorage.getItem(ACTIVE_TAB_STORAGE_KEY)
        if (stored && tabs.some((tab) => tab.id === stored)) {
          return stored
        }
      }

      return tabs[0].id
    })
  }, [status, tabs])

  useEffect(() => {
    if (status !== 'ready' || !activeTab || typeof window === 'undefined') {
      return
    }
    window.localStorage.setItem(ACTIVE_TAB_STORAGE_KEY, activeTab)
  }, [activeTab, status])

  const activeSection = sections.find((section) => section.id === activeTab)

  return (
    <main className="app">
      <Hero status={status} backend={backend} error={error} />

      {status === 'ready' ? (
        tabs.length > 0 && activeTab ? (
          <div className="tab-container">
            <TabBar tabs={tabs} activeTab={activeTab} onSelect={setActiveTab} />

            <div className="tab-content">
              {activeSection ? (
                <Section key={activeSection.id} section={activeSection} />
              ) : activeTab === DOCS_TAB.id ? (
                <DocsSection groups={documentationGroups} />
              ) : (
                <div className="empty-tab">请选择一个模块查看示例。</div>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-tab">暂无可用模块，请稍后重试。</div>
        )
      ) : (
        <Placeholder status={status} error={error} />
      )}

      <Footer />
    </main>
  )
}

export default App
