const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function submitGame(youtubeUrl, label) {
  const res = await fetch(`${API_URL}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ youtube_url: youtubeUrl, label: label || undefined }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Server error ${res.status}`)
  }
  return res.json()
}

export async function fetchStatus(jobId) {
  const res = await fetch(`${API_URL}/status/${jobId}`)
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`)
  return res.json()
}

export async function fetchResults(jobId) {
  const res = await fetch(`${API_URL}/results/${jobId}`)
  if (!res.ok) throw new Error(`Results fetch failed: ${res.status}`)
  return res.json()
}

export async function deleteJob(jobId) {
  const res = await fetch(`${API_URL}/jobs/${jobId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
  return res.json()
}

/**
 * Stream chat response via SSE. Calls onChunk(text) for each token.
 * Returns an abort controller so the caller can cancel.
 */
export function streamChat(jobId, messages, onChunk, onDone, onError) {
  const controller = new AbortController()

  fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, messages }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.text()
        throw new Error(err || `Chat error ${res.status}`)
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        // Parse SSE lines
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line in buffer
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              onDone?.()
              return
            }
            onChunk(data)
          }
        }
      }
      onDone?.()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError?.(err)
      }
    })

  return controller
}
