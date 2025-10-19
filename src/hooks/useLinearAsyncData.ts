import { useEffect, useState } from 'react'
import {
  Matrix,
  conv2d,
  gaussianBlur,
  matmulAsync,
  sobelFilter,
} from '@jayce789/numjs'
import {
  CONV_INPUT_VALUES,
  CONV_KERNEL_VALUES,
  SOBEL_INPUT_VALUES,
  MATMUL_LEFT_VALUES,
  MATMUL_RIGHT_VALUES,
} from '../data/demoConstants'
import type { LinearAsyncResult, Status } from '../types'

export const useLinearAsyncData = (status: Status) => {
  const [data, setData] = useState<LinearAsyncResult>({})

  useEffect(() => {
    if (status !== 'ready') {
      setData({})
      return
    }

    let cancelled = false

    const run = async () => {
      const results: LinearAsyncResult = {}

      try {
        const left = new Matrix(MATMUL_LEFT_VALUES, 2, 3)
        const right = new Matrix(MATMUL_RIGHT_VALUES, 3, 2)
        results.matmulAsync = await matmulAsync(left, right)
      } catch (err) {
        results.matmulAsyncError =
          err instanceof Error ? err.message : 'matmulAsync 不受当前后端支持'
      }

      try {
        const input = new Matrix(CONV_INPUT_VALUES, 3, 3)
        const kernel = new Matrix(CONV_KERNEL_VALUES, 2, 2)
        results.conv2d = await conv2d(input, kernel)
      } catch (err) {
        results.conv2dError =
          err instanceof Error ? err.message : 'conv2d 不受当前后端支持'
      }

      try {
        const input = new Matrix(SOBEL_INPUT_VALUES, 3, 3)
        const response = await sobelFilter(input, { magnitude: true })
        results.sobel = response
      } catch (err) {
        results.sobelError =
          err instanceof Error ? err.message : 'sobelFilter 不受当前后端支持'
      }

      try {
        const input = new Matrix(SOBEL_INPUT_VALUES, 3, 3)
        results.gaussian = await gaussianBlur(input, { sigma: 1.2, size: 3 })
      } catch (err) {
        results.gaussianError =
          err instanceof Error
            ? err.message
            : 'gaussianBlur 不受当前后端支持'
      }

      if (!cancelled) {
        setData(results)
      }
    }

    run()

    return () => {
      cancelled = true
    }
  }, [status])

  return data
}
