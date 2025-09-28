import React, { useState } from 'react'
import { useLocation } from 'react-router-dom'
import STLViewer from '../components/STLViewer'
import { absoluteApiUrl } from '../lib/api'
import Tabs from '../components/Tabs'
import ChatbotPanel from '../components/ChatbotPanel'

export default function Results() {
  const { state } = useLocation() as any
  const lungsUrlAbs = absoluteApiUrl(state?.lungs_url as string | null)
  const airwayUrlAbs = absoluteApiUrl(state?.airway_url as string | null)

  const [activeTab, setActiveTab] = useState<'render' | 'deposition'>('render')
  const [simulating, setSimulating] = useState(false)
  const [simDone, setSimDone] = useState(false)

  async function runSimulation() {
    if (simulating) return
    setSimulating(true)
    await new Promise(res => setTimeout(res, 1200))
    setSimulating(false)
    setSimDone(true)
    setActiveTab('deposition')
  }

  return (
    <div style={{ padding: 16 }}>
      <Tabs active={activeTab} onChange={setActiveTab} onSimulate={runSimulation} />

      {activeTab === 'render' && (
        <div className="panel" style={{ height: '80vh', padding: 0 }}>
          {airwayUrlAbs ? (
            <STLViewer src={airwayUrlAbs} />
          ) : (
            lungsUrlAbs && <STLViewer src={lungsUrlAbs} />
          )}
        </div>
      )}

      {activeTab === 'deposition' && (
        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 16 }}>
          <ChatbotPanel />
          <div className="panel" style={{ height: '80vh' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ marginTop: 0 }}>Deposition graphs</h3>
              <button className="btn secondary" disabled={simulating} onClick={runSimulation}>{simulating ? 'Running…' : 'Re-run Simulation'}</button>
            </div>
            {simDone ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ background: 'rgba(18,24,38,0.35)', border: '1px solid #22304a', borderRadius: 10, height: 280, display: 'grid', placeItems: 'center' }}>Bar chart placeholder</div>
                <div style={{ background: 'rgba(18,24,38,0.35)', border: '1px solid #22304a', borderRadius: 10, height: 280, display: 'grid', placeItems: 'center' }}>Line chart placeholder</div>
                <div style={{ background: 'rgba(18,24,38,0.35)', border: '1px solid #22304a', borderRadius: 10, height: 280, display: 'grid', placeItems: 'center' }}>Lobe-wise heatmap placeholder</div>
                <div style={{ background: 'rgba(18,24,38,0.35)', border: '1px solid #22304a', borderRadius: 10, height: 280, display: 'grid', placeItems: 'center' }}>Dose-response placeholder</div>
              </div>
            ) : (
              <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: '#a7b3c2' }}>
                <div>{simulating ? 'Running simulation…' : 'Click Run Simulation to generate deposition graphs.'}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}


