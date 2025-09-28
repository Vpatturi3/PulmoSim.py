import React, { useMemo, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import STLViewer from '../components/STLViewer'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function absolutize(url?: string | null): string | null {
  if (!url) return null
  if (url.startsWith('http://') || url.startsWith('https://')) return url
  return `${API_BASE}${url}`
}

function STLMesh({ url, color = '#87CEEB', opacity = 1 }: { url: string, color?: string, opacity?: number }) {
  const geomRef = useRef<THREE.BufferGeometry>(null)
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)

  useMemo(async () => {
    if (!url) return
    const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js')
    const loader = new STLLoader()
    loader.load(url, (geom: THREE.BufferGeometry) => setGeometry(geom))
  }, [url])

  if (!geometry) return null
  return (
    <mesh>
      <meshStandardMaterial color={color} opacity={opacity} transparent={opacity < 1} />
      <primitive object={geometry} attach="geometry" ref={geomRef} />
    </mesh>
  )
}

export default function Results() {
  const { state } = useLocation() as any
  const lungsUrlAbs = absolutize(state?.lungs_url)
  const airwayUrlAbs = absolutize(state?.airway_url)
  const meta: any = state?.meta

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>Results</h2>
        <Link className="btn" to="/">← Back to Home</Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '420px 1fr', gap: 24, marginTop: 16 }}>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Downloads</h3>
          {lungsUrlAbs ? (
            <p><a className="btn" href={lungsUrlAbs} download>Download lungs.stl</a></p>
          ) : (
            <p>No lungs STL found.</p>
          )}
          {airwayUrlAbs ? (
            <p><a className="btn secondary" href={airwayUrlAbs} download>Download airway.stl</a></p>
          ) : (
            <p>No airway STL.</p>
          )}
          <h3>Processing details</h3>
          <pre style={{ margin: 0 }}>{JSON.stringify(meta || {}, null, 2)}</pre>
        </div>

        <div className="panel" style={{ height: '75vh', padding: 0 }}>
          {lungsUrlAbs && <STLViewer src={lungsUrlAbs} />}
        </div>
      </div>
    </div>
  )
}


