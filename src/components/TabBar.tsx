export type TabDefinition = {
  id: string
  label: string
  type: 'section' | 'docs'
}

type TabBarProps = {
  tabs: TabDefinition[]
  activeTab: string | null
  onSelect: (tabId: string) => void
}

export const TabBar = ({ tabs, activeTab, onSelect }: TabBarProps) => (
  <nav className="tab-bar">
    {tabs.map((tab) => (
      <button
        key={tab.id}
        type="button"
        className={`tab-button ${activeTab === tab.id ? 'is-active' : ''}`}
        onClick={() => onSelect(tab.id)}
      >
        {tab.label}
      </button>
    ))}
  </nav>
)
