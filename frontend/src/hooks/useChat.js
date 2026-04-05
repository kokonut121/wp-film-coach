import { useState, useCallback, useRef } from 'react'
import { streamChat } from '../api'

export function useChat(jobId) {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const controllerRef = useRef(null)

  const sendMessage = useCallback(
    (text) => {
      if (!text.trim() || !jobId || isStreaming) return

      const userMsg = { role: 'user', content: text.trim() }
      const newMessages = [...messages, userMsg]
      setMessages([...newMessages, { role: 'assistant', content: '' }])
      setIsStreaming(true)

      controllerRef.current = streamChat(
        jobId,
        newMessages,
        (chunk) => {
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            updated[updated.length - 1] = { ...last, content: last.content + chunk }
            return updated
          })
        },
        () => setIsStreaming(false),
        (err) => {
          setIsStreaming(false)
          setMessages((prev) => {
            const updated = [...prev]
            updated[updated.length - 1] = {
              role: 'assistant',
              content: `Error: ${err.message}`,
            }
            return updated
          })
        }
      )
    },
    [jobId, messages, isStreaming]
  )

  const cancelStream = useCallback(() => {
    controllerRef.current?.abort()
    setIsStreaming(false)
  }, [])

  return { messages, isStreaming, sendMessage, cancelStream }
}
