import React from 'react'

type TabKey = 'render' | 'deposition'

export default function Tabs({ active, onChange, onSimulate }: {
  active: TabKey
  onChange: (key: TabKey) => void
  onSimulate: () => void
}) {
  return (
    <div className="tabbar">
      <div className="tabs">
        <button
          className={`tab ${active === 'render' ? 'active' : ''}`}
          onClick={() => onChange('render')}
        >3D Render</button>
        <button
          className={`tab ${active === 'deposition' ? 'active' : ''}`}
          onClick={() => onChange('deposition')}
        >Deposition Graphs</button>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn secondary" onClick={onSimulate}>Run Simulation</button>
        <a className="btn" href="/" style={{ textDecoration: 'none' }}>← Back to Home</a>
      </div>
    </div>
  )
}


