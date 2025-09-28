import React, { useEffect } from 'react'
import STLViewer from '../components/STLViewer'
import { getApiBase } from '../lib/api'

export default function DemoViewer() {
  const params = new URLSearchParams(window.location.search)
  const inside = params.get('inside') === '1'
  useEffect(() => {
    // Make the iframe page itself transparent (no dark rectangle)
    document.documentElement.style.background = 'transparent'
    document.body.style.background = 'transparent'
    document.body.style.margin = '0'
  }, [])
  return (
    <div style={{ width: '100%', height: '100vh', background: 'transparent' }}>
      <STLViewer src={`${getApiBase()}/demo/longen`} />
      {inside && (
        <div style={{ position: 'absolute', top: 16, left: 16, background: 'rgba(0,0,0,0.4)', padding: 8, borderRadius: 8 }}>
          <span>Looking inside (clip preview)</span>
        </div>
      )}
    </div>
  )
}


