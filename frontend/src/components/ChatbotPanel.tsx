import React, { useEffect, useRef, useState } from 'react'

type ChatMessage = { role: 'user' | 'assistant', content: string }

export default function ChatbotPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hi! I\'m your PulmoSim assistant. Ask about deposition results, therapy options, or how the model works.' }
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function onSend(e?: React.FormEvent) {
    e?.preventDefault()
    if (!input.trim() || busy) return
    const userMsg: ChatMessage = { role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setBusy(true)

    // Placeholder: simulate Gemini response. Integrate backend later.
    await new Promise(res => setTimeout(res, 900))
    const assistant: ChatMessage = {
      role: 'assistant',
      content: 'This is a placeholder Gemini response. Once the simulation backend returns deposition metrics, I\'ll summarize hotspots, compare lobe-level deposition, and suggest next steps.'
    }
    setMessages(prev => [...prev, assistant])
    setBusy(false)
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
            placeholder="Ask about deposition or treatment..."
            value={input}
            onChange={e => setInput(e.target.value)}
          />
          <button className="btn" type="submit" disabled={busy}>{busy ? 'Sending…' : 'Send'}</button>
        </form>
      </div>
    </div>
  )
}


