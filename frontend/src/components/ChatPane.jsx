import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { useChat } from '../hooks/useChat'

export default function ChatPane({ jobId, report }) {
  const { messages, isStreaming, sendMessage } = useChat(jobId)
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const [showReport, setShowReport] = useState(true)

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isStreaming])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim()) return
    sendMessage(input)
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-panel" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
        padding: '6px 14px', borderBottom: '1px solid var(--border)',
      }}>
        {report && (
          <button
            className="metrics-tab"
            style={{ padding: '3px 8px', fontSize: '0.65rem', borderBottom: 'none' }}
            onClick={() => setShowReport((s) => !s)}
          >
            {showReport ? 'Hide Report' : 'Show Report'}
          </button>
        )}
      </div>
      <div className="chat-messages">
        {/* Auto-generated report */}
        {report && showReport && (
          <div className="report-block">
            <ReactMarkdown>{report}</ReactMarkdown>
          </div>
        )}

        {/* Chat messages */}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            {msg.role === 'assistant' ? (
              <ReactMarkdown>{msg.content || (isStreaming && i === messages.length - 1 ? '...' : '')}</ReactMarkdown>
            ) : (
              msg.content
            )}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask about the game..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? '...' : 'Send'}
        </button>
      </form>
    </div>
  )
}
