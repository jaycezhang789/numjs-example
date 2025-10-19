import type { Matrix } from '@jayce789/numjs'
import { clone2D, matrixFrom2D, matrixTo2D } from './matrixHelpers'

export const safeTake = (
  matrix: Matrix,
  axis: number,
  indices: readonly number[],
) => {
  try {
    return matrix.take(axis, indices)
  } catch {
    const base = matrixTo2D(matrix)
    const normalized = Array.from(indices)

    if (axis === 0) {
      const rows = normalized
        .map((index) => base[index])
        .filter((row): row is number[] => Array.isArray(row))
        .map((row) => row.slice())
      return matrixFrom2D(rows)
    }

    if (axis === 1) {
      const rows = base.map((row) =>
        normalized
          .map((col) => row[col])
          .filter((value): value is number => typeof value === 'number'),
      )
      return matrixFrom2D(rows)
    }

    return matrixFrom2D([])
  }
}

export const safePut = (
  matrix: Matrix,
  axis: number,
  indices: readonly number[],
  values: Matrix,
) => {
  try {
    return matrix.put(axis, indices, values)
  } catch {
    const base = clone2D(matrixTo2D(matrix))
    const valData = matrixTo2D(values)
    const normalized = Array.from(indices)

    if (axis === 0) {
      normalized.forEach((rowIndex, offset) => {
        if (rowIndex >= 0 && rowIndex < base.length) {
          base[rowIndex] = valData[offset]?.slice() ?? base[rowIndex]
        }
      })
    } else if (axis === 1) {
      for (let r = 0; r < base.length; r += 1) {
        const valueRow = valData[r] ?? []
        normalized.forEach((colIndex, offset) => {
          if (
            colIndex >= 0 &&
            colIndex < base[r].length &&
            offset < valueRow.length
          ) {
            base[r][colIndex] = valueRow[offset]
          }
        })
      }
    }

    return matrixFrom2D(base)
  }
}

export const safeGather = (
  matrix: Matrix,
  rowIndices: readonly number[],
  colIndices: readonly number[],
) => {
  try {
    return matrix.gather(rowIndices, colIndices)
  } catch {
    const base = matrixTo2D(matrix)
    const rows = rowIndices.map((row) =>
      colIndices
        .map((col) => base[row]?.[col])
        .filter((value): value is number => typeof value === 'number'),
    )
    return matrixFrom2D(rows)
  }
}

export const safeGatherPairs = (
  matrix: Matrix,
  rowIndices: readonly number[],
  colIndices: readonly number[],
) => {
  try {
    return matrix.gatherPairs(rowIndices, colIndices)
  } catch {
    const base = matrixTo2D(matrix)
    const values = rowIndices.map((row, index) => {
      const col = colIndices[index]
      return base[row]?.[col] ?? 0
    })
    return matrixFrom2D([values])
  }
}

export const safeScatter = (
  matrix: Matrix,
  rowIndices: readonly number[],
  colIndices: readonly number[],
  values: Matrix,
) => {
  try {
    return matrix.scatter(rowIndices, colIndices, values)
  } catch {
    const base = clone2D(matrixTo2D(matrix))
    const valData = matrixTo2D(values)

    rowIndices.forEach((row, rIdx) => {
      const rowData = valData[rIdx] ?? []
      colIndices.forEach((col, cIdx) => {
        if (
          row >= 0 &&
          row < base.length &&
          col >= 0 &&
          col < base[row].length &&
          cIdx < rowData.length
        ) {
          base[row][col] = rowData[cIdx]
        }
      })
    })

    return matrixFrom2D(base)
  }
}

export const safeScatterPairs = (
  matrix: Matrix,
  rowIndices: readonly number[],
  colIndices: readonly number[],
  values: Matrix,
) => {
  try {
    return matrix.scatterPairs(rowIndices, colIndices, values)
  } catch {
    const base = clone2D(matrixTo2D(matrix))
    const flatValues = matrixTo2D(values).flat()

    rowIndices.forEach((row, index) => {
      const col = colIndices[index]
      if (
        row >= 0 &&
        row < base.length &&
        col >= 0 &&
        col < base[row].length &&
        index < flatValues.length
      ) {
        base[row][col] = flatValues[index]
      }
    })

    return matrixFrom2D(base)
  }
}
