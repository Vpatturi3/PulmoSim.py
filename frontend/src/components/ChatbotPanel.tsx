import React, { useEffect, useRef, useState } from 'react'

type ChatMessage = { role: 'user' | 'assistant', content: string }

export default function ChatbotPanel({ jobId }: { jobId?: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hi! I\'m your PulmoSim assistant. Ask about deposition results, therapy options, or how the model works.' }
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function onSend(e?: React.FormEvent) {
    e?.preventDefault()
    const text = input.trim()
    if (!text || busy) return
    const userMsg: ChatMessage = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setBusy(true)

    try {
      if (!jobId) throw new Error('Missing job id')
      const form = new FormData()
      form.append('job_id', jobId)
      if (text) form.append('q', text)
      const resp = await fetch('/chatbot_answer', { method: 'POST', body: form })
      if (!resp.ok) {
        let message = `HTTP ${resp.status}`
        try { const j = await resp.json(); if (j?.error) message = j.error } catch {}
        throw new Error(message)
      }
      const json = await resp.json()
      const assistant: ChatMessage = {
        role: 'assistant',
        content: json.message || 'Generated analysis.'
      }
      setMessages(prev => [...prev, assistant])
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message || String(err)}` }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="sidebar">
      <div className="panel" style={{ padding: 12, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <h3 style={{ marginTop: 0 }}>Assistant</h3>
        <div style={{ flex: 1, overflowY: 'auto', paddingRight: 6 }}>
          {messages.map((m, i) => (
            <div key={i} style={{
              margin: '8px 0',
              background: m.role === 'assistant' ? 'rgba(18,24,38,0.6)' : 'transparent',
              border: '1px solid #22304a',
              borderRadius: 10,
              padding: 10
            }}>
              <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>{m.role === 'assistant' ? 'Assistant' : 'You'}</div>
              <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form onSubmit={onSend} style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 8, marginTop: 8 }}>
          <input
            placeholder="Ask which inhaler worked best..."
            value={input}
            onChange={e => setInput(e.target.value)}
          />
          <button className="btn" type="submit" disabled={busy}>{busy ? 'Sending…' : 'Send'}</button>
        </form>
      </div>
    </div>
  )
}


