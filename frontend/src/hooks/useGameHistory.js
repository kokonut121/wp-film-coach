import { useState, useCallback } from 'react'

const STORAGE_KEY = 'wp-film-coach-history'

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []
  } catch {
    return []
  }
}

function save(games) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(games))
}

export function useGameHistory() {
  const [games, setGames] = useState(load)

  const addGame = useCallback((game) => {
    setGames((prev) => {
      const next = [game, ...prev.filter((g) => g.job_id !== game.job_id)]
      save(next)
      return next
    })
  }, [])

  const removeGame = useCallback((jobId) => {
    setGames((prev) => {
      const next = prev.filter((g) => g.job_id !== jobId)
      save(next)
      return next
    })
  }, [])

  const updateGame = useCallback((jobId, updates) => {
    setGames((prev) => {
      const next = prev.map((g) => (g.job_id === jobId ? { ...g, ...updates } : g))
      save(next)
      return next
    })
  }, [])

  return { games, addGame, removeGame, updateGame }
}
