import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import STLViewer from '../components/STLViewer'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function LungsLogo() {
  return (
    <div className="logo" aria-label="PulmoSim logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 12c0-4-1.5-7-3-8-2 2-5 6-5 10 0 3 2 5 5 5 2 0 3-1 3-3V12z"></path>
        <path d="M12 12c0-4 1.5-7 3-8 2 2 5 6 5 10 0 3-2 5-5 5-2 0-3-1-3-3V12z"></path>
        <path d="M12 3v6"></path>
      </svg>
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const uploadRef = useRef<HTMLDivElement | null>(null)
  const [seeInside, setSeeInside] = useState(false)
  const [airwaysVisible, setAirwaysVisible] = useState(false)
  const [airflowVisible, setAirflowVisible] = useState(false)

  useEffect(() => {
    if (inputRef.current) {
      // Enable folder selection for browsers that support it
      // @ts-ignore
      inputRef.current.webkitdirectory = true
      // @ts-ignore
      inputRef.current.directory = true
      inputRef.current.multiple = true
    }
  }, [])

  const readAllEntries = useCallback((reader: any): Promise<any[]> => {
    return new Promise(resolve => {
      const entries: any[] = []
      const readBatch = () => reader.readEntries((batch: any[]) => {
        if (!batch.length) resolve(entries)
        else { entries.push(...batch); readBatch() }
      })
      readBatch()
    })
  }, [])

  const traverseEntry = useCallback(async (entry: any, out: File[]) => {
    if (!entry) return
    if (entry.isFile) {
      await new Promise<void>(res => entry.file((file: File) => { out.push(file); res() }))
      return
    }
    if (entry.isDirectory) {
      const reader = entry.createReader()
      const entries = await readAllEntries(reader)
      for (const e of entries) {
        // eslint-disable-next-line no-await-in-loop
        await traverseEntry(e, out)
      }
    }
  }, [readAllEntries])

  const scrollToUpload = () => uploadRef.current?.scrollIntoView({ behavior: 'smooth' })

  const onDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const dt = e.dataTransfer
    const list: File[] = []
    if (dt.items && dt.items.length) {
      // @ts-ignore
      const items: DataTransferItem[] = Array.from(dt.items)
      for (const it of items) {
        const entry = (it as any).webkitGetAsEntry?.() || (it as any).getAsEntry?.()
        if (entry) {
          // eslint-disable-next-line no-await-in-loop
          await traverseEntry(entry, list)
        } else {
          const file = it.getAsFile()
          if (file) list.push(file)
        }
      }
    } else if (dt.files && dt.files.length) {
      list.push(...Array.from(dt.files))
    }
    setFiles(list)
  }, [traverseEntry])

  const onSubmit = useCallback(async () => {
    if (files.length === 0) { setError('Drop a DICOM folder/zip or files'); return }
    setBusy(true); setError(null)
    try {
      const form = new FormData()
      files.forEach(f => form.append('files', f))
      form.append('iso', '1.0')
      form.append('lung_hu_low', '-1000')
      form.append('lung_hu_high', '-400')
      form.append('decimate', '0.9')
      form.append('airway_enabled', 'false')
      const resp = await fetch(`${API_BASE}/process`, { method: 'POST', body: form })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      navigate('/results', { state: json })
    } catch (err: any) {
      setError(err.message || String(err))
    } finally {
      setBusy(false)
    }
  }, [files, navigate])

  const toggleSeeInside = () => setSeeInside(v => !v)
  const toggleAirways = () => setAirwaysVisible(v => !v)
  const toggleAirflow = () => setAirflowVisible(v => !v)

  return (
    <div>
      <section className="hero">
        <div>
          <div className="brand">
            <LungsLogo />
            <div>
              <div className="title">PulmoSim</div>
              <div className="subtitle">CT to interactive lungs – visualize, simulate, and compare therapies.</div>
            </div>
          </div>
          <div className="hud">
            <button className="btn" onClick={scrollToUpload}>Analyze your lungs</button>
            <a
              className="btn secondary"
              href="https://github.com/Vpatturi3/PulmoSim.py.git"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub repository"
              style={{ width: 44, height: 44, display: 'grid', placeItems: 'center', padding: 0 }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.52 2.87 8.34 6.84 9.69.5.1.68-.22.68-.48 0-.24-.01-.87-.01-1.71-2.78.61-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1.01.07 1.54 1.06 1.54 1.06.9 1.57 2.36 1.12 2.94.86.09-.67.35-1.12.64-1.38-2.22-.26-4.55-1.14-4.55-5.09 0-1.12.39-2.04 1.03-2.76-.1-.26-.45-1.32.1-2.75 0 0 .84-.27 2.75 1.05A9.3 9.3 0 0 1 12 7.1c.85 0 1.7.12 2.5.35 1.9-1.32 2.74-1.05 2.74-1.05.55 1.43.2 2.49.1 2.75.64.72 1.03 1.64 1.03 2.76 0 3.96-2.34 4.82-4.57 5.08.36.32.69.95.69 1.92 0 1.39-.01 2.51-.01 2.85 0 .26.18.58.69.48A10.07 10.07 0 0 0 22 12.26C22 6.58 17.52 2 12 2z"/>
              </svg>
            </a>
          </div>
        </div>
        <div style={{ minHeight: 520 }}>
          <div className="viewer-panel" style={{ height: 720, borderRadius: 16, overflow: 'hidden', background: 'transparent', position: 'relative' }}>
            <div onClick={toggleSeeInside} style={{ position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none' }} />
            <STLViewer
              src={`${API_BASE}/demo/longen`}
              airwaySrc={undefined}
              seeInside={seeInside}
              airwaysVisible={airwaysVisible}
              airflowVisible={airflowVisible}
              onToggleInside={toggleSeeInside}
            />
          </div>
        </div>
      </section>

      <section className="overview">
        <div className="info-panel">
          <div className="info-header">
            <div className="spark" aria-hidden="true" />
            <h2 style={{ margin: 0 }}>What PulmoSim does</h2>
            <p className="lede">CT to interactive lungs – visualize, simulate, and compare therapies.</p>
          </div>
          <div className="info-grid">
            <div className="info-item">
              <div className="info-icon" />
              <div>
                <div className="info-title">3D reconstruction</div>
                <div className="info-sub">Turn DICOM/NIfTI into accurate lung meshes you can explore.</div>
              </div>
            </div>
            <div className="info-item">
              <div className="info-icon" />
              <div>
                <div className="info-title">AI recommendations</div>
                <div className="info-sub">Personalized therapy suggestions based on your scan.</div>
              </div>
            </div>
            <div className="info-item">
              <div className="info-icon" />
              <div>
                <div className="info-title">Visualize responses</div>
                <div className="info-sub">See airflow and airway changes with different medications.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section ref={uploadRef} className="upload-section" id="upload">
        <div className="drop-panel">
          <div className="drop-surface"
               onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }}
               onDrop={onDrop}
               onClick={() => inputRef.current?.click()}>
            <div className="drop-inner">
              <div className="drop-title">Select files to upload</div>
              <div className="drop-sub">Drag & Drop a DICOM folder, ZIP series, or NIfTI (.nii/.nii.gz)</div>
              {files.length > 0 && (
                <div className="drop-count">{files.length} file(s) selected</div>
              )}
            </div>
          </div>
          <input id="fileInput" ref={inputRef} type="file" accept=".zip,.nii,.nii.gz,.dcm" multiple style={{ display: 'none' }} onChange={e => setFiles(Array.from(e.target.files || []))} />
          <div className="drop-actions">
            <button className="btn" disabled={busy} onClick={onSubmit}>{busy ? 'Processing…' : 'Process'}</button>
          </div>
          {error && <div style={{ color: '#ff8a8a', marginTop: 8 }}>{error}</div>}
        </div>
        
      </section>
    </div>
  )
}


