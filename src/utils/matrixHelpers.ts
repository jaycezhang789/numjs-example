import type { Matrix } from '@jayce789/numjs'

export const truncate = (value: string, max = 80) =>
  value.length > max ? `${value.slice(0, max)}…` : value

export const formatNumber = (value: number): string => {
  if (!Number.isFinite(value)) {
    return value.toString()
  }

  const abs = Math.abs(value)
  if ((abs > 0 && abs < 0.001) || abs >= 10_000) {
    return value.toExponential(2)
  }

  return value.toFixed(4).replace(/\.?0+$/, '')
}

export const formatFloatArray = (array: ArrayLike<number>) =>
  Array.from(array)
    .map((value) => formatNumber(value))
    .join(', ')

const toNumeric = (value: number | bigint | boolean): number => {
  if (typeof value === 'number') return value
  if (typeof value === 'bigint') return Number(value)
  return value ? 1 : 0
}

export const matrixToTable = (matrix: Matrix): string[][] => {
  const { rows, cols, dtype } = matrix
  const values = Array.from(matrix.toArray() as Iterable<number | bigint>)
  const table: string[][] = []

  for (let r = 0; r < rows; r += 1) {
    const row: string[] = []
    for (let c = 0; c < cols; c += 1) {
      const raw = values[r * cols + c]
      if (dtype === 'bool') {
        row.push(Number(raw) > 0 ? 'true' : 'false')
      } else if (typeof raw === 'bigint') {
        row.push(raw.toString())
      } else {
        row.push(formatNumber(Number(raw)))
      }
    }
    table.push(row)
  }

  return table
}

export const matrixTo2D = (matrix: Matrix): number[][] => {
  const rows = matrix.rows
  const cols = matrix.cols
  const raw = matrix.toArray() as ArrayLike<number | bigint | boolean>
  const result: number[][] = []

  for (let r = 0; r < rows; r += 1) {
    const row: number[] = []
    for (let c = 0; c < cols; c += 1) {
      row.push(toNumeric(raw[r * cols + c]))
    }
    result.push(row)
  }

  return result
}

export const matrixFrom2D = (data: number[][]): Matrix => {
  const rows = data.length
  const cols = rows > 0 ? data[0].length : 0
  return new Matrix(data.flat(), rows, cols)
}

export const clone2D = (data: number[][]): number[][] =>
  data.map((row) => row.slice())
