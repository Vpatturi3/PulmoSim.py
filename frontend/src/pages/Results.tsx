import React, { useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import STLViewer from '../components/STLViewer'
import { absoluteApiUrl } from '../lib/api'
import Tabs from '../components/Tabs'
import ChatbotPanel from '../components/ChatbotPanel'

export default function Results() {
  const { state } = useLocation() as any
  const lungsUrlAbs = absoluteApiUrl(state?.lungs_url as string | null)
  const airwayUrlAbs = absoluteApiUrl(state?.airway_url as string | null)
  const jobId = state?.job_id as string | undefined

  const [activeTab, setActiveTab] = useState<'render' | 'deposition'>('render')
  const [simulating, setSimulating] = useState(false)
  const [simDone, setSimDone] = useState(false)
  const [images, setImages] = useState<{ [k: string]: string } | null>(null)
  const [bestId, setBestId] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [ragBusy, setRagBusy] = useState(false)
  const [ragSummary, setRagSummary] = useState<string | null>(null)

  const friendlyName: Record<string, string> = {
    mdi: 'Metered Dose Inhaler',
    dpi: 'Dry Powder Inhaler',
    neb: 'Nebulizer',
  }

  const imageList = useMemo(() => {
    if (!images) return [] as { id: string, url: string }[]
    return Object.entries(images).map(([id, url]) => ({ id, url: absoluteApiUrl(url)! }))
  }, [images])

  async function runSimulation() {
    if (simulating) return
    if (!jobId) { setError('Missing job id'); return }
    setSimulating(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('job_id', jobId)
      const resp = await fetch('/simulate_deposition', { method: 'POST', body: form })
      if (!resp.ok) {
        let message = `HTTP ${resp.status}`
        try { const j = await resp.json(); if (j?.error) message = j.error } catch {}
        throw new Error(message)
      }
      const json = await resp.json()
      setImages(json.images)
      setBestId(json.best ?? null)
      setMetrics(json.metrics ?? null)
      setSimDone(true)
      setActiveTab('deposition')
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setSimulating(false)
    }
  }

  async function ensureRag() {
    if (!jobId || !images || ragBusy) return
    setRagBusy(true)
    try {
      const form = new FormData()
      form.append('job_id', jobId)
      // Chatbot endpoint will auto-run RAG if diagnosis.json missing
      const resp = await fetch('/chatbot_answer', { method: 'POST', body: form })
      if (resp.ok) {
        const json = await resp.json()
        setRagSummary(json?.message || null)
      }
    } catch {}
    finally { setRagBusy(false) }
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
          <ChatbotPanel jobId={jobId} />
          <div className="panel" style={{ height: '80vh', overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ marginTop: 0 }}>Deposition results</h3>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn secondary" disabled={simulating} onClick={runSimulation}>{simulating ? 'Running…' : 'Re-run Simulation'}</button>
              </div>
            </div>
            {error && <div style={{ color: '#ff8a8a', marginBottom: 8 }}>{error}</div>}
            {simDone && images ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                  {imageList.map(img => {
                    const label = friendlyName[img.id] || img.id.toUpperCase()
                    const isBest = bestId === img.id
                    const score = metrics?.[img.id]?.total as number | undefined
                    return (
                      <div key={img.id} style={{ background: 'rgba(18,24,38,0.35)', border: '1px solid #22304a', borderRadius: 12, padding: 10 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                          <div style={{ fontWeight: 600 }}>{label}</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            {typeof score === 'number' && (
                              <span style={{ fontSize: 12, opacity: 0.8 }}>Score: {score.toFixed(2)}</span>
                            )}
                            {isBest && (
                              <span style={{ fontSize: 12, padding: '2px 8px', borderRadius: 999, background: 'linear-gradient(90deg,#2dd4bf,#22c55e)', color: '#0b1323' }}>Best</span>
                            )}
                          </div>
                        </div>
                        <div style={{ background: '#0b1323', borderRadius: 8, overflow: 'hidden' }}>
                          <img src={img.url} alt={label} style={{ display: 'block', width: '100%', height: 'auto' }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
                {ragSummary && (
                  <div style={{ marginTop: 12, background: 'rgba(18,24,38,0.35)', border: '1px solid #22304a', borderRadius: 12, padding: 12 }}>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>RAG Assessment</div>
                    <div style={{ whiteSpace: 'pre-wrap' }}>{ragSummary}</div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: '#a7b3c2' }}>
                <div>{simulating ? 'Running simulation…' : 'Click Run Simulation to generate deposition images.'}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}


