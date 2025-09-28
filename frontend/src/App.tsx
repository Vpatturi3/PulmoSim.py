import React, { useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Center, useProgress, Html } from '@react-three/drei'
import * as THREE from 'three'
import { getApiBase } from './lib/api'

function Loader() {
  const { progress } = useProgress()
  return <Html center>{progress.toFixed(0)} % loaded</Html>
}

function STLMesh({ url, color = '#87CEEB', opacity = 1 }: { url: string, color?: string, opacity?: number }) {
  const geomRef = useRef<THREE.BufferGeometry>(null)
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)

  useMemo(async () => {
    if (!url) return
    const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js')
    const loader = new STLLoader()
    loader.load(url, (geom: THREE.BufferGeometry) => {
      setGeometry(geom)
    })
  }, [url])

  if (!geometry) return null
  return (
    <mesh>
      <meshStandardMaterial color={color} opacity={opacity} transparent={opacity < 1} />
      <primitive object={geometry} attach="geometry" ref={geomRef} />
    </mesh>
  )
}

export default function App() {
  const [files, setFiles] = useState<FileList | null>(null)
  const [iso, setIso] = useState(1.0)
  const [huLow, setHuLow] = useState(-1000)
  const [huHigh, setHuHigh] = useState(-400)
  const [decimate, setDecimate] = useState(0.5)
  const [airwayEnabled, setAirwayEnabled] = useState(false)
  const [seed, setSeed] = useState({ z: 10, y: 100, x: 100 })

  const [lungsUrl, setLungsUrl] = useState<string | null>(null)
  const [airwayUrl, setAirwayUrl] = useState<string | null>(null)
  const [meta, setMeta] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    setLungsUrl(null)
    setAirwayUrl(null)

    try {
      const form = new FormData()
      if (!files || files.length === 0) {
        setError('Please select files')
        setBusy(false)
        return
      }
      Array.from(files).forEach(f => form.append('files', f))
      form.append('iso', String(iso))
      form.append('lung_hu_low', String(huLow))
      form.append('lung_hu_high', String(huHigh))
      form.append('decimate', String(decimate))
      form.append('airway_enabled', String(airwayEnabled))
      if (airwayEnabled) {
        form.append('airway_seed_z', String(seed.z))
        form.append('airway_seed_y', String(seed.y))
        form.append('airway_seed_x', String(seed.x))
      }

      const resp = await fetch(`${getApiBase()}/process`, { method: 'POST', body: form })
      if (!resp.ok) {
        let message = `HTTP ${resp.status}`
        try {
          const errJson = await resp.json()
          if (errJson?.error) message = errJson.error
        } catch (_) {
          try { message = await resp.text() } catch (_) {}
        }
        throw new Error(message)
      }
      const json = await resp.json()
      setLungsUrl(`${getApiBase()}${json.lungs_url}`)
      setAirwayUrl(json.airway_url ? `${getApiBase()}${json.airway_url}` : null)
      setMeta(json.meta)
    } catch (err: any) {
      setError(err.message || String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', height: '100vh' }}>
      <aside style={{ padding: 16, borderRight: '1px solid #eee', overflowY: 'auto' }}>
        <h2>PulmoSim</h2>
        <form onSubmit={onSubmit}>
          <div>
            <label>Upload DICOM (.zip or .dcm) or NIfTI (.nii/.nii.gz)</label>
            <input type="file" multiple onChange={e => setFiles(e.target.files)} />
          </div>

          <h3>Preprocessing</h3>
          <div>
            <label>Isotropic spacing (mm)</label>
            <input type="number" step="0.1" min={0.5} max={3} value={iso} onChange={e => setIso(parseFloat(e.target.value))} />
          </div>
          <div>
            <label>HU low</label>
            <input type="number" value={huLow} onChange={e => setHuLow(parseInt(e.target.value))} />
          </div>
          <div>
            <label>HU high</label>
            <input type="number" value={huHigh} onChange={e => setHuHigh(parseInt(e.target.value))} />
          </div>
          <div>
            <label>Decimate</label>
            <input type="range" min={0.1} max={0.95} step={0.05} value={decimate} onChange={e => setDecimate(parseFloat(e.target.value))} /> {decimate.toFixed(2)}
          </div>

          <h3>Airway</h3>
          <div>
            <label>
              <input type="checkbox" checked={airwayEnabled} onChange={e => setAirwayEnabled(e.target.checked)} /> Enable airway
            </label>
          </div>
          {airwayEnabled && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              <div>
                <label>Seed z</label>
                <input type="number" value={seed.z} onChange={e => setSeed({ ...seed, z: parseInt(e.target.value) })} />
              </div>
              <div>
                <label>Seed y</label>
                <input type="number" value={seed.y} onChange={e => setSeed({ ...seed, y: parseInt(e.target.value) })} />
              </div>
              <div>
                <label>Seed x</label>
                <input type="number" value={seed.x} onChange={e => setSeed({ ...seed, x: parseInt(e.target.value) })} />
              </div>
            </div>
          )}

          <div style={{ marginTop: 12 }}>
            <button type="submit" disabled={busy}>{busy ? 'Processing…' : 'Process'}</button>
          </div>
        </form>

        {error && <div style={{ color: 'red' }}>{error}</div>}
        {meta && (
          <pre style={{ background: '#f7f7f7', padding: 8, borderRadius: 6 }}>{JSON.stringify(meta, null, 2)}</pre>
        )}
        {lungsUrl && (
          <div style={{ marginTop: 12 }}>
            <a href={lungsUrl} download>Download lungs.stl</a>
          </div>
        )}
        {airwayUrl && (
          <div>
            <a href={airwayUrl} download>Download airway.stl</a>
          </div>
        )}
      </aside>

      <main>
        <Canvas camera={{ position: [150, 150, 150], near: 0.1, far: 5000 }} shadows>
          <ambientLight intensity={0.7} />
          <directionalLight position={[100, 100, 100]} intensity={0.6} />
          <Center>
            {lungsUrl && <STLMesh url={lungsUrl} />}
            {airwayUrl && <STLMesh url={airwayUrl} color="#FF8C00" opacity={0.6} />}
          </Center>
          <OrbitControls makeDefault />
        </Canvas>
      </main>
    </div>
  )
}


