import { useState, useEffect, useRef } from 'react'
import { fetchStatus } from '../api'

export function useJobStatus(jobId) {
  const [status, setStatus] = useState({ stage: 'queued', pct: 0 })
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (!jobId) return

    const poll = async () => {
      try {
        const data = await fetchStatus(jobId)
        setStatus(data)
        if (data.stage === 'done' || data.stage === 'error') {
          clearInterval(intervalRef.current)
          if (data.stage === 'error') {
            setError(data.error_message || 'Processing failed')
          }
        }
      } catch (err) {
        setError(err.message)
        clearInterval(intervalRef.current)
      }
    }

    poll()
    intervalRef.current = setInterval(poll, 3000)

    return () => clearInterval(intervalRef.current)
  }, [jobId])

  return { ...status, error }
}
