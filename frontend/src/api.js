const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const LOCAL_PROXY_URL = 'http://127.0.0.1:8001'
const UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024

export async function submitGame(youtubeUrl, label) {
  // Route YouTube downloads through the local proxy to avoid IP blocking
  const res = await fetch(`${LOCAL_PROXY_URL}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ youtube_url: youtubeUrl, label: label || undefined }),
  }).catch(() => null)

  if (!res || !res.ok) {
    if (!res) {
      throw new Error('Local proxy not running. Start it with: python local_proxy.py')
    }
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || err.detail || `Server error ${res.status}`)
  }
  const data = await res.json()
  if (data.error) throw new Error(data.error)
  return data
}

async function parseError(res) {
  const err = await res.json().catch(() => ({}))
  throw new Error(err.detail || err.error || `Server error ${res.status}`)
}

export async function uploadGame(file, label, onProgress) {
  const initRes = await fetch(`${API_URL}/uploads/init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename: file.name,
      label: label || undefined,
      content_type: file.type || 'application/octet-stream',
      total_size: file.size,
    }),
  })

  if (!initRes.ok) {
    await parseError(initRes)
  }

  const { job_id } = await initRes.json()
  const totalChunks = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK_SIZE))

  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * UPLOAD_CHUNK_SIZE
    const end = Math.min(file.size, start + UPLOAD_CHUNK_SIZE)
    const chunk = file.slice(start, end)
    const url = new URL(`${API_URL}/uploads/${job_id}/chunk`)
    url.searchParams.set('index', String(index))
    url.searchParams.set('total_chunks', String(totalChunks))
    url.searchParams.set('start_byte', String(start))

    const chunkRes = await fetch(url, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/octet-stream',
      },
      body: chunk,
    })

    if (!chunkRes.ok) {
      await parseError(chunkRes)
    }

    onProgress?.(Math.round((end / file.size) * 100))
  }

  const completeRes = await fetch(`${API_URL}/uploads/${job_id}/complete`, {
    method: 'POST',
  })

  if (!completeRes.ok) {
    await parseError(completeRes)
  }

  return completeRes.json()
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
