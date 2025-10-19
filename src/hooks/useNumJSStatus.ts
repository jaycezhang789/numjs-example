import { useEffect, useState } from 'react'
import { backendKind, init } from '@jayce789/numjs'
import type { Status } from '../types'

export const useNumJSStatus = () => {
  const [status, setStatus] = useState<Status>('loading')
  const [backend, setBackend] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const bootstrap = async () => {
      try {
        await init()
        if (cancelled) return

        setBackend(backendKind())
        setStatus('ready')
      } catch (err) {
        if (cancelled) return

        const message =
          err instanceof Error ? err.message : '无法初始化 @jayce789/numjs'
        setError(message)
        setStatus('error')
      }
    }

    bootstrap()

    return () => {
      cancelled = true
    }
  }, [])

  return { status, backend, error }
}
