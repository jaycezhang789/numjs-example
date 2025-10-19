import { useEffect, useMemo, useState } from 'react'
import {
  Matrix,
  add,
  backendKind,
  broadcastTo,
  conv2d,
  div,
  dot,
  im2col,
  init,
  matrixFromFixed,
  matmul,
  matmulAsync,
  maxPool,
  median,
  mul,
  nanmean,
  nansum,
  neg,
  quantile,
  percentile,
  compress,
  sobelFilter,
  gaussianBlur,
  svd,
  qr,
  solve,
  eigen,
  sub,
  sumUnsafe,
  sum,
  where,
  avgPool,
} from '@jayce789/numjs'
import './App.css'

type Status = 'loading' | 'ready' | 'error'

type Table = string[][]

type DemoMatrix = {
  label: string
  table: Table
}

type DemoOutput = {
  label: string
  table?: Table
  scalar?: string
}

type DemoBlock = {
  title: string
  description: string
  expression: string
  inputs: DemoMatrix[]
  outputs: DemoOutput[]
  highlight?: string
}

type DemoSection = {
  id: string
  heading: string
  summary: string
  demos: DemoBlock[]
}

type DocGroup = {
  title: string
  description: string
  links: {
    label: string
    url: string
    note?: string
  }[]
}

type LinearAsyncData = {
  matmulAsync?: Matrix
  matmulAsyncError?: string
  conv2d?: Matrix
  conv2dError?: string
  sobel?: {
    gx: Matrix
    gy: Matrix
    magnitude?: Matrix
  }
  sobelError?: string
  gaussian?: Matrix
  gaussianError?: string
}

const truncate = (value: string, max = 80) =>
  value.length > max ? `${value.slice(0, max)}…` : value

const toNumeric = (value: number | bigint | boolean): number => {
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'bigint') {
    return Number(value)
  }
  return value ? 1 : 0
}

const formatFloatArray = (array: ArrayLike<number>) =>
  Array.from(array)
    .map((value) => formatNumber(value))
    .join(', ')

const matrixTo2D = (matrix: Matrix): number[][] => {
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

const matrixFrom2D = (data: number[][]): Matrix => {
  const rows = data.length
  const cols = rows > 0 ? data[0].length : 0
  return new Matrix(data.flat(), rows, cols)
}

const clone2D = (data: number[][]): number[][] => data.map((row) => row.slice())

const MATMUL_LEFT_VALUES = [1, 0, -1, 2, 3, 1]
const MATMUL_RIGHT_VALUES = [2, 1, 0, -1, 1, 2]
const CONV_INPUT_VALUES = [1, 2, 1, 0, 1, 0, 2, 1, 2]
const CONV_KERNEL_VALUES = [1, 0, -1, 1]
const POOL_INPUT_VALUES = [
  1, 2, 3, 4,
  5, 6, 7, 8,
  9, 10, 11, 12,
  13, 14, 15, 16,
]
const SOBEL_INPUT_VALUES = [
  1, 2, 1,
  0, 0, 0,
  -1, -2, -1,
]
const SVD_INPUT_VALUES = [
  4, 0, 3,
  5, -1, 0,
  2, 3, 1,
]
const QR_INPUT_VALUES = [
  12, -51, 4,
  6, 167, -68,
  -4, 24, -41,
]
const SOLVE_MATRIX_VALUES = [
  3, 2,
  1, 4,
]
const SOLVE_B_VALUES = [7, 10]
const EIGEN_INPUT_VALUES = [
  2, -1,
  -1, 2,
]

const safeTake = (matrix: Matrix, axis: number, indices: readonly number[]) => {
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

const safePut = (
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
          if (colIndex >= 0 && colIndex < base[r].length && offset < valueRow.length) {
            base[r][colIndex] = valueRow[offset]
          }
        })
      }
    }

    return matrixFrom2D(base)
  }
}

const safeGather = (
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

const safeGatherPairs = (
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

const safeScatter = (
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

const safeScatterPairs = (
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

const formatNumber = (value: number): string => {
  if (!Number.isFinite(value)) {
    return value.toString()
  }

  const abs = Math.abs(value)
  if ((abs > 0 && abs < 0.001) || abs >= 10_000) {
    return value.toExponential(2)
  }

  return value.toFixed(4).replace(/\.?0+$/, '')
}

const matrixToTable = (matrix: Matrix): Table => {
  const { rows, cols, dtype } = matrix
  const values = Array.from(matrix.toArray() as Iterable<number | bigint>)
  const table: Table = []

  for (let r = 0; r < rows; r++) {
    const row: string[] = []
    for (let c = 0; c < cols; c++) {
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

const useNumJSStatus = () => {
  const [status, setStatus] = useState<Status>('loading')
  const [backend, setBackend] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const bootstrap = async () => {
      try {
        await init()
        if (cancelled) {
          return
        }

        setBackend(backendKind())
        setStatus('ready')
      } catch (err) {
        if (cancelled) {
          return
        }

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

const documentationGroups: DocGroup[] = [
  {
    title: '矩阵与广播',
    description:
      '快速浏览 Matrix 类的基础方法、广播规则以及与 NumPy 的对照说明。',
    links: [
      {
        label: 'Quick Start & 常用操作',
        url: 'https://github.com/jaycezhang789/numjs#quick-start',
      },
      {
        label: '从 NumPy 迁移指南',
        url: 'https://github.com/jaycezhang789/numjs/blob/main/docs/tutorials/from-numpy-migration.md',
      },
    ],
  },
  {
    title: '线性代数与分解',
    description:
      '了解 matmul、svd、qr、solve 等高阶线性代数能力及底层后端差异。',
    links: [
      {
        label: '线性代数 API（TypeScript 定义）',
        url: 'https://github.com/jaycezhang789/numjs/blob/main/dist/index.d.ts#L293',
        note: '通过类型声明查看完整矩阵方法与函数签名',
      },
      {
        label: '后端架构与性能要点',
        url: 'https://github.com/jaycezhang789/numjs/blob/main/docs/tutorials/backends.md',
      },
    ],
  },
  {
    title: '统计与聚合',
    description:
      'sum、nanmean、dot 等归约函数的数值稳定性策略及使用建议。',
    links: [
      {
        label: '稳定归约与数值策略',
        url: 'https://github.com/jaycezhang789/numjs#stable-reductions',
      },
      {
        label: '数值输出控制（round/toOutputArray）',
        url: 'https://github.com/jaycezhang789/numjs#output--rounding-display-only',
      },
    ],
  },
  {
    title: '扩展生态',
    description:
      '数据帧、Arrow/Polars、WASM/WebGPU 等高级能力的入口与教程。',
    links: [
      {
        label: 'WASM / WebGPU 加速',
        url: 'https://github.com/jaycezhang789/numjs/blob/main/docs/tutorials/webgpu.md',
      },
      {
        label: '交互式 Playground',
        url: 'https://github.com/jaycezhang789/numjs/tree/main/docs/interactive',
      },
      {
        label: '未来规划与扩展路线',
        url: 'https://github.com/jaycezhang789/numjs/blob/main/docs/future.md',
      },
    ],
  },
]

const useLinearAsyncData = (status: Status) => {
  const [data, setData] = useState<LinearAsyncData>({})

  useEffect(() => {
    if (status !== 'ready') {
      setData({})
      return
    }

    let cancelled = false

    const run = async () => {
      const results: LinearAsyncData = {}

      try {
        const left = new Matrix(MATMUL_LEFT_VALUES, 2, 3)
        const right = new Matrix(MATMUL_RIGHT_VALUES, 3, 2)
        results.matmulAsync = await matmulAsync(left, right)
      } catch (err) {
        results.matmulAsyncError =
          err instanceof Error
            ? err.message
            : 'matmulAsync 不受当前后端支持'
      }

      try {
        const input = new Matrix(CONV_INPUT_VALUES, 3, 3)
        const kernel = new Matrix(CONV_KERNEL_VALUES, 2, 2)
        results.conv2d = await conv2d(input, kernel)
      } catch (err) {
        results.conv2dError =
          err instanceof Error
            ? err.message
            : 'conv2d 不受当前后端支持'
      }

      try {
        const input = new Matrix(SOBEL_INPUT_VALUES, 3, 3)
        const response = await sobelFilter(input, { magnitude: true })
        results.sobel = response
      } catch (err) {
        results.sobelError =
          err instanceof Error
            ? err.message
            : 'sobelFilter 不受当前后端支持'
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

const useDemoSections = (status: Status, asyncData: LinearAsyncData) =>
  useMemo<DemoSection[]>(() => {
    if (status !== 'ready') {
      return []
    }

    const baseConstructMatrix = new Matrix([1, 2, 3, 4], 2, 2)
    const lazySource = baseConstructMatrix.toLazy()
    const fromLazyMatrix = Matrix.fromLazy(lazySource)
    const fixedMatrix = matrixFromFixed([1234n, -5678n, 910n, 112n], 2, 2, 2)
    const toLazySummary = `LazyArray(${baseConstructMatrix.rows}×${baseConstructMatrix.cols})`

    const bytesSeed = new Float64Array([9, 8, 7, 6])
    const bytesConstructMatrix = Matrix.fromBytes(
      new Uint8Array(bytesSeed.buffer.slice(0)),
      2,
      2,
      'float64',
    )
    const fromHandleMatrix = Matrix.fromHandle(
      (bytesConstructMatrix as any)._handle,
    )
    const fromHandleWithDTypeMatrix = Matrix.fromHandleWithDType(
      (fixedMatrix as any)._handle,
      'fixed64',
      { fixedScale: fixedMatrix.fixedScale ?? undefined },
    )

    const elementwiseA = new Matrix([1, 3, 5, 7], 2, 2)
    const elementwiseB = new Matrix([2, 4, 6, 8], 2, 2)
    const elementwiseAdd = add(elementwiseA, elementwiseB)
    const elementwiseSub = sub(elementwiseA, elementwiseB)
    const elementwiseMul = mul(elementwiseA, elementwiseB)
    const elementwiseDiv = div(elementwiseA, elementwiseB)
    const elementwiseNeg = neg(elementwiseA)

    const concatTop = new Matrix([1, 2, 3, 4], 2, 2)
    const concatBottom = new Matrix([5, 6, 7, 8], 2, 2)
    const concatVertical = concatTop.concat(concatBottom)
    const concatHorizontal = concatTop.concat(concatBottom, 1)
    const stackAxis0 = concatTop.stack(concatBottom)
    const stackAxis1 = concatTop.stack(concatBottom, 1)

    const rawValues = new Matrix([3.2, -1.5, 4.7, 0.3, 6.9, -2.1], 2, 3)
    const clipped = rawValues.clip(0, 5)
    const rounded = rawValues.round(1)
    const ints = clipped.astype('int32')

    const matmulLeft = new Matrix(MATMUL_LEFT_VALUES, 2, 3)
    const matmulRight = new Matrix(MATMUL_RIGHT_VALUES, 3, 2)
    const matmulResult = matmul(matmulLeft, matmulRight)

    const viewMatrix = new Matrix([1, 2, 3, 4, 5, 6], 2, 3)
    const transposed = viewMatrix.transpose()
    const secondRow = viewMatrix.row(1)
    const firstColumn = viewMatrix.column(0)
    const leadingSlice = viewMatrix.slice(
      { start: 0, end: 2 },
      { start: 0, end: 2 },
    )
    const trailingSlice = viewMatrix.slice(
      { start: 0, end: 2 },
      { start: 1, end: 3 },
    )
    const dtypeInfoSummary = (() => {
      const info = viewMatrix.dtypeInfo
      return `size=${info.size}, kind=${info.kind}, float=${info.isFloat}, signed=${info.isSigned}`
    })()
    const toArrayPreview = Array.from(
      viewMatrix.toArray() as Iterable<number | bigint>,
    ).join(', ')
    const toLazyStructureSummary = `LazyArray(${viewMatrix.rows}×${viewMatrix.cols})`

    const rowVector = new Matrix([1.5, -0.5, 0.25], 1, 3)
    const broadcastedRow = broadcastTo(rowVector, 3, 3)
    const baseGrid = new Matrix([1, 4, 7, 2, 5, 8, 3, 6, 9], 3, 3)
    const shiftedGrid = add(baseGrid, broadcastedRow)

    const columnVector = new Matrix([2, -1, 0], 3, 1)
    const broadcastedColumn = broadcastTo(columnVector, 3, 3)
    const columnShifted = add(baseGrid, broadcastedColumn)

    const mask = new Matrix(
      [
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        true,
        false,
      ],
      3,
      3,
      { dtype: 'bool' },
    )
    const zerosGrid = new Matrix(Array(9).fill(0), 3, 3)
    const maskedGrid = where(mask, shiftedGrid, zerosGrid)
    const compressedValues = compress(mask, shiftedGrid)

    const statsMatrix = new Matrix([6, 4, 2, NaN, 8, 10], 2, 3)
    const totalSum = formatNumber(Number(sum(statsMatrix).toArray()[0]))
    const meanValue = formatNumber(Number(nanmean(statsMatrix).toArray()[0]))
    const nanSumValue = formatNumber(Number(nansum(statsMatrix).toArray()[0]))
    const medianValue = formatNumber(Number(median(statsMatrix).toArray()[0]))
    const quantile75 = formatNumber(
      Number(quantile(statsMatrix, 0.75).toArray()[0]),
    )
    const percentile90 = formatNumber(
      Number(percentile(statsMatrix, 90).toArray()[0]),
    )
    const sumUnsafeValue = formatNumber(sumUnsafe(statsMatrix))

    const convInputMatrix = new Matrix(CONV_INPUT_VALUES, 3, 3)
    const convKernelMatrix = new Matrix(CONV_KERNEL_VALUES, 2, 2)
    const im2colMatrix = im2col(convInputMatrix, 2, 2)
    const poolInputMatrix = new Matrix(POOL_INPUT_VALUES, 4, 4)
    const maxPoolMatrix = maxPool(poolInputMatrix, 2, 2, { stride: 2 })
    const avgPoolMatrix = avgPool(poolInputMatrix, 2, 2, { stride: 2 })
    const sobelInputMatrix = new Matrix(SOBEL_INPUT_VALUES, 3, 3)

    const matmulAsyncOutputs: DemoOutput[] = asyncData.matmulAsync
      ? [
          {
            label: '异步结果 (2×2)',
            table: matrixToTable(asyncData.matmulAsync),
          },
        ]
      : [
          {
            label: '异步结果',
            scalar: asyncData.matmulAsyncError ?? '正在计算…',
          },
        ]

    const convOutputs: DemoOutput[] = [
      asyncData.conv2d
        ? {
            label: 'conv2d 输入卷积 (2×2)',
            table: matrixToTable(asyncData.conv2d),
          }
        : {
            label: 'conv2d 输入卷积',
            scalar: asyncData.conv2dError ?? '正在计算…',
          },
      {
        label: 'im2col(input, 2, 2)',
        table: matrixToTable(im2colMatrix),
      },
      {
        label: 'maxPool(kernel=2, stride=2)',
        table: matrixToTable(maxPoolMatrix),
      },
      {
        label: 'avgPool(kernel=2, stride=2)',
        table: matrixToTable(avgPoolMatrix),
      },
    ]

    const sobelBlurOutputs: DemoOutput[] = []
    if (asyncData.sobel) {
      sobelBlurOutputs.push(
        {
          label: 'sobelFilter gx',
          table: matrixToTable(asyncData.sobel.gx),
        },
        {
          label: 'sobelFilter gy',
          table: matrixToTable(asyncData.sobel.gy),
        },
      )
      if (asyncData.sobel.magnitude) {
        sobelBlurOutputs.push({
          label: '梯度幅值',
          table: matrixToTable(asyncData.sobel.magnitude),
        })
      }
    } else {
      sobelBlurOutputs.push({
        label: 'sobelFilter',
        scalar: asyncData.sobelError ?? '正在计算…',
      })
    }

    sobelBlurOutputs.push(
      asyncData.gaussian
        ? {
            label: 'gaussianBlur (σ=1.2)',
            table: matrixToTable(asyncData.gaussian),
          }
        : {
            label: 'gaussianBlur',
            scalar: asyncData.gaussianError ?? '正在计算…',
          },
    )

    const svdInputMatrix = new Matrix(SVD_INPUT_VALUES, 3, 3)
    let svdOutputs: DemoOutput[]
    try {
      const { u, s, vt } = svd(svdInputMatrix)
      svdOutputs = [
        { label: 'U', table: matrixToTable(u) },
        { label: 'Σ (对角)', scalar: formatFloatArray(s) },
        { label: 'Vᵀ', table: matrixToTable(vt) },
      ]
    } catch (err) {
      svdOutputs = [
        {
          label: 'svd',
          scalar:
            err instanceof Error ? err.message : 'svd 不受当前后端支持',
        },
      ]
    }

    const qrInputMatrix = new Matrix(QR_INPUT_VALUES, 3, 3)
    let qrOutputs: DemoOutput[]
    try {
      const { q, r } = qr(qrInputMatrix)
      qrOutputs = [
        { label: 'Q', table: matrixToTable(q) },
        { label: 'R', table: matrixToTable(r) },
      ]
    } catch (err) {
      qrOutputs = [
        {
          label: 'qr',
          scalar:
            err instanceof Error ? err.message : 'qr 不受当前后端支持',
        },
      ]
    }

    const solveMatrixA = new Matrix(SOLVE_MATRIX_VALUES, 2, 2)
    const solveMatrixB = new Matrix(SOLVE_B_VALUES, 2, 1)
    let solveOutputs: DemoOutput[]
    try {
      const solution = solve(solveMatrixA, solveMatrixB)
      solveOutputs = [
        { label: '解向量', table: matrixToTable(solution) },
      ]
    } catch (err) {
      solveOutputs = [
        {
          label: 'solve',
          scalar:
            err instanceof Error ? err.message : 'solve 不受当前后端支持',
        },
      ]
    }

    const eigenInputMatrix = new Matrix(EIGEN_INPUT_VALUES, 2, 2)
    let eigenOutputs: DemoOutput[]
    try {
      const { values, vectors } = eigen(eigenInputMatrix)
      eigenOutputs = [
        { label: '特征值', scalar: formatFloatArray(values) },
        { label: '特征向量', table: matrixToTable(vectors) },
      ]
    } catch (err) {
      eigenOutputs = [
        {
          label: 'eigen',
          scalar:
            err instanceof Error ? err.message : 'eigen 不受当前后端支持',
        },
      ]
    }

    const dotVectorA = new Matrix([1, 2, 3], 1, 3)
    const dotVectorB = new Matrix([4, 5, 6], 1, 3)
    const dotValue = formatNumber(
      Number(dot(dotVectorA, dotVectorB).toArray()[0]),
    )

    const indexMatrix = new Matrix([10, 20, 30, 40, 50, 60], 2, 3)
    const takeColumns = safeTake(indexMatrix, 1, [0, 2])
    const putRow = safePut(
      indexMatrix,
      0,
      [1],
      new Matrix([700, 800, 900], 1, 3),
    )
    const gatherSubmatrix = safeGather(indexMatrix, [0, 1], [0, 2])
    const gatherPairsResult = safeGatherPairs(indexMatrix, [0, 1, 1], [0, 1, 2])
    const scatterResult = safeScatter(
      indexMatrix,
      [0],
      [1, 2],
      new Matrix([111, 222], 1, 2),
    )
    const scatterPairsResult = safeScatterPairs(
      indexMatrix,
      [0, 1, 1],
      [0, 1, 2],
      new Matrix([900, 901, 902], 1, 3),
    )

    const serializationMatrix = new Matrix([1.2345, -2.5, 3.75, 4], 2, 2)
    const bytesBuffer = serializationMatrix.toBytes()
    const restoredFromBytes = Matrix.fromBytes(
      bytesBuffer,
      serializationMatrix.rows,
      serializationMatrix.cols,
      serializationMatrix.dtype,
    )
    const bytesPreview = Array.from(bytesBuffer.slice(0, 8)).join(', ')
    const outputStringsPreview = (
      serializationMatrix.toOutputArray({
        as: 'string',
        decimals: 2,
      }) as (string | number | bigint)[]
    )
      .map(String)
      .join(', ')
    const jsonPreview = truncate(JSON.stringify(serializationMatrix.toJSON()))
    const toStringPreview = truncate(serializationMatrix.toString())

    return [
      {
        id: 'basics',
        heading: '矩阵基础',
        summary: '展示 NumJS 的矩阵构造、逐元素运算与索引写入 API。',
        demos: [
          {
            title: '多样构造方式',
            description:
              '展示 new Matrix、Matrix.fromLazy、Matrix.fromBytes 以及 Matrix.fromHandle* 等基础构造手段。',
            expression:
              'new Matrix(data, rows, cols) · Matrix.fromLazy(lazy) · Matrix.fromBytes(bytes, rows, cols, dtype) · Matrix.fromHandle(handle)',
            inputs: [
              { label: 'new Matrix 源', table: matrixToTable(baseConstructMatrix) },
            ],
            outputs: [
              {
                label: 'new Matrix',
                table: matrixToTable(baseConstructMatrix),
              },
              { label: 'Matrix.fromLazy', table: matrixToTable(fromLazyMatrix) },
              { label: 'Matrix.fromFixed(scale=2)', table: matrixToTable(fixedMatrix) },
              {
                label: 'Matrix.fromBytes',
                table: matrixToTable(bytesConstructMatrix),
              },
              { label: 'Matrix.fromHandle', table: matrixToTable(fromHandleMatrix) },
              {
                label: 'Matrix.fromHandleWithDType',
                table: matrixToTable(fromHandleWithDTypeMatrix),
              },
              { label: 'toLazy()', scalar: toLazySummary },
            ],
            highlight:
              '日常使用推荐 new Matrix/Matrix.fromBytes/fromFixed/fromLazy；Matrix.fromHandle* 可在自定义原生扩展里复用后端句柄。',
          },
          {
            title: '结构信息与视图',
            description:
              'rows/cols/dtype 等 getter 以及 toArray()/toLazy()/row()/column()/slice() 的基础用法。',
            expression:
              'matrix.rows, matrix.dtype, matrix.dtypeInfo, matrix.toArray(), matrix.row(i), matrix.slice(...)',
            inputs: [
              { label: '示例矩阵 (2×3)', table: matrixToTable(viewMatrix) },
            ],
            outputs: [
              {
                label: 'rows × cols',
                scalar: `${viewMatrix.rows} × ${viewMatrix.cols}`,
              },
              { label: 'dtype', scalar: viewMatrix.dtype },
              { label: 'fixedScale', scalar: String(viewMatrix.fixedScale) },
              { label: 'dtypeInfo', scalar: dtypeInfoSummary },
              { label: 'toArray()', scalar: toArrayPreview },
              { label: 'toLazy()', scalar: toLazyStructureSummary },
              { label: 'row(1)', table: matrixToTable(secondRow) },
              { label: 'column(0)', table: matrixToTable(firstColumn) },
              {
                label: 'slice(0:2, 0:2)',
                table: matrixToTable(leadingSlice),
              },
            ],
            highlight:
              '结构 getter 不触发复制；toArray() 返回底层 TypedArray，toLazy() 则生成可组合的延迟视图；row/column/slice 返回新的矩阵视图。',
          },
          {
            title: '逐元素运算',
            description:
              'add、sub、mul 分别执行矩阵的逐元素加、减、乘。',
            expression: 'add(A, B) · sub(A, B) · mul(A, B)',
            inputs: [
              { label: '矩阵 A', table: matrixToTable(elementwiseA) },
              { label: '矩阵 B', table: matrixToTable(elementwiseB) },
            ],
            outputs: [
              { label: 'A + B', table: matrixToTable(elementwiseAdd) },
              { label: 'A - B', table: matrixToTable(elementwiseSub) },
              { label: 'A ⊙ B', table: matrixToTable(elementwiseMul) },
            ],
            highlight: 'mul 对应 Hadamard 乘法，返回与输入同形的矩阵。',
          },
          {
            title: '除法与取反',
            description: 'div(a, b) 执行逐元素除法，neg(a) 返回取反矩阵。',
            expression: 'div(A, B), neg(A)',
            inputs: [
              { label: '矩阵 A', table: matrixToTable(elementwiseA) },
              { label: '矩阵 B', table: matrixToTable(elementwiseB) },
            ],
            outputs: [
              { label: 'A ÷ B', table: matrixToTable(elementwiseDiv) },
              { label: '-A', table: matrixToTable(elementwiseNeg) },
            ],
            highlight:
              '除法结果保留浮点精度，neg 常用于构造差分或梯度方向。',
          },
          {
            title: '矩阵拼接',
            description: 'Matrix.concat 支持按行或按列拼接相同形状的矩阵。',
            expression: 'matrix.concat(other, axis?)',
            inputs: [
              { label: '矩阵 A (2×2)', table: matrixToTable(concatTop) },
              { label: '矩阵 B (2×2)', table: matrixToTable(concatBottom) },
            ],
            outputs: [
              {
                label: '按行拼接 (4×2)',
                table: matrixToTable(concatVertical),
              },
              {
                label: '按列拼接 (2×4)',
                table: matrixToTable(concatHorizontal),
              },
            ],
            highlight:
              'axis 缺省为 0 纵向堆叠；传入 1 时横向拼接，常用于扩展特征列。',
          },
          {
            title: '矩阵堆叠',
            description:
              'stack 在指定 axis 上新增维度堆叠矩阵，与 concat 的连接方式互补。',
            expression: 'matrix.stack(other, axis?)',
            inputs: [
              { label: '矩阵 A (2×2)', table: matrixToTable(concatTop) },
              { label: '矩阵 B (2×2)', table: matrixToTable(concatBottom) },
            ],
            outputs: [
              { label: 'stack(axis=0)', table: matrixToTable(stackAxis0) },
              { label: 'stack(axis=1)', table: matrixToTable(stackAxis1) },
            ],
            highlight:
              'stack 可在不同 axis 上生成深度或通道维度，适合构建批数据或张量。',
          },
          {
            title: '矩阵转置',
            description: 'transpose() 将矩阵的行列互换，返回新的转置矩阵。',
            expression: 'matrix.transpose()',
            inputs: [
              { label: '原始矩阵 (2×3)', table: matrixToTable(viewMatrix) },
            ],
            outputs: [{ label: '转置矩阵 (3×2)', table: matrixToTable(transposed) }],
            highlight:
              '转置操作不会修改原矩阵，常用于线性代数与特征处理场景。',
          },
          {
            title: '广播示例',
            description:
              'broadcastTo(rows, cols) 可将较小矩阵扩展为目标形状以便对齐运算。',
            expression: 'matrix.broadcastTo(rows, cols)',
            inputs: [
              { label: '行向量 (1×3)', table: matrixToTable(rowVector) },
            ],
            outputs: [
              {
                label: 'broadcastTo(3, 3)',
                table: matrixToTable(broadcastedRow),
              },
            ],
            highlight:
              'broadcastTo 会复制行/列以满足目标形状，是进行批量偏移/拼接的常见前置步骤。',
          },
          {
            title: '索引与写入',
            description:
              'take/put/gather/scatter 提供灵活的索引读取与写入操作。',
            expression:
              'matrix.take(axis, indices) · matrix.put(axis, indices, values) · matrix.gather(...) · matrix.scatter(...)',
            inputs: [
              { label: '原矩阵 (2×3)', table: matrixToTable(indexMatrix) },
            ],
            outputs: [
              {
                label: 'take(axis=1, [0,2])',
                table: matrixToTable(takeColumns),
              },
              {
                label: 'put(axis=0, row=1)',
                table: matrixToTable(putRow),
              },
              {
                label: 'gather([0,1],[0,2])',
                table: matrixToTable(gatherSubmatrix),
              },
              {
                label: 'gatherPairs',
                table: matrixToTable(gatherPairsResult),
              },
              {
                label: 'scatter row0 cols[1,2]',
                table: matrixToTable(scatterResult),
              },
              {
                label: 'scatterPairs',
                table: matrixToTable(scatterPairsResult),
              },
            ],
            highlight:
              '这些方法会优先调用内建 API；若当前后端缺少实现，示例会自动回退到 JS 版本，仍然在不修改原矩阵的前提下演示稀疏更新与索引。',
          },
          {
            title: '数值规范化',
            description:
              'clip、round 与 astype 组合使用可进行限幅、格式化与整数化。',
            expression: 'matrix.clip(0, 5).round(1).astype("int32")',
            inputs: [{ label: '原始数据', table: matrixToTable(rawValues) }],
            outputs: [
              { label: 'clip [0, 5]', table: matrixToTable(clipped) },
              { label: 'round(1)', table: matrixToTable(rounded) },
              { label: 'astype("int32")', table: matrixToTable(ints) },
            ],
            highlight:
              'clip 保障区间，round 控制展示精度，astype 便于后续与整型接口对接。',
          },
          {
            title: '序列化与输出',
            description:
              'toBytes、toOutputArray、toString、toJSON 便于持久化与调试展示。',
            expression:
              'matrix.toBytes() · matrix.toOutputArray() · matrix.toString() · Matrix.fromBytes(...)',
            inputs: [
              { label: '示例矩阵', table: matrixToTable(serializationMatrix) },
            ],
            outputs: [
              { label: 'toBytes() 前 8 字节', scalar: bytesPreview },
              {
                label: 'toOutputArray(as="string")',
                scalar: outputStringsPreview,
              },
              {
                label: 'Matrix.fromBytes(...)',
                table: matrixToTable(restoredFromBytes),
              },
              { label: 'toString()', scalar: toStringPreview },
              { label: 'toJSON()', scalar: jsonPreview },
            ],
            highlight:
              'toOutputArray 支持字符串/数值/BigInt 格式；配合 Matrix.fromBytes 可进行二进制序列化，toString/toJSON 则适合调试输出。',
          },
        ],
      },
      {
        id: 'linear-algebra',
        heading: '线性代数',
        summary: '矩阵乘法、转置与行列视图是线性代数的核心操作。',
        demos: [
          {
            title: '矩阵乘法',
            description: 'matmul(a, b) 执行标准的矩阵乘法。',
            expression: 'matmul(A, B)',
            inputs: [
              { label: '矩阵 A (2×3)', table: matrixToTable(matmulLeft) },
              { label: '矩阵 B (3×2)', table: matrixToTable(matmulRight) },
            ],
            outputs: [
              { label: 'A × B', table: matrixToTable(matmulResult) },
            ],
            highlight:
              'NumJS 会在 N-API 与 WebAssembly 后端之间自动选择最优实现。',
          },
          {
            title: '异步矩阵乘法',
            description: 'matmulAsync(a, b, options?) 提供异步矩阵乘法，优先利用 GPU/N-API。',
            expression: 'await matmulAsync(A, B)',
            inputs: [
              { label: '矩阵 A (2×3)', table: matrixToTable(matmulLeft) },
              { label: '矩阵 B (3×2)', table: matrixToTable(matmulRight) },
            ],
            outputs: matmulAsyncOutputs,
            highlight:
              '若当前后端无 GPU 支持，会自动退回 WASM/JS 实现；示例会提示实际运行状态。',
          },
          {
            title: '转置与切片',
            description: 'Matrix 提供 transpose、row、column 等常用视图操作。',
            expression: 'matrix.transpose(), matrix.row(1), matrix.column(0)',
            inputs: [
              { label: '原始矩阵 (2×3)', table: matrixToTable(viewMatrix) },
            ],
            outputs: [
              { label: '转置矩阵 (3×2)', table: matrixToTable(transposed) },
              { label: '第 2 行 (1×3)', table: matrixToTable(secondRow) },
              { label: '第 1 列 (2×1)', table: matrixToTable(firstColumn) },
              {
                label: '左上角切片 (2×2)',
                table: matrixToTable(leadingSlice),
              },
              {
                label: '右侧切片 (2×2)',
                table: matrixToTable(trailingSlice),
              },
            ],
            highlight:
              '行列视图返回新的矩阵实例，可继续参与计算；slice 支持按区间截取子矩阵。',
          },
          {
            title: '卷积与池化',
            description:
              'conv2d、im2col、maxPool、avgPool 演示卷积与池化的基础流程。',
            expression:
              'await conv2d(input, kernel) · im2col(input, 2, 2) · maxPool(input, 2, 2, stride=2) · avgPool(...)',
            inputs: [
              { label: '卷积输入 (3×3)', table: matrixToTable(convInputMatrix) },
              { label: '卷积核 (2×2)', table: matrixToTable(convKernelMatrix) },
              { label: '池化输入 (4×4)', table: matrixToTable(poolInputMatrix) },
            ],
            outputs: convOutputs,
            highlight:
              'conv2d 在部分浏览器可能退回 JS 版本；im2col 展示卷积展开原理，max/avgPool 则演示 2×2 stride=2 的池化结果。',
          },
          {
            title: '边缘检测与模糊',
            description:
              'sobelFilter 与 gaussianBlur 常用于提取梯度和图像平滑。',
            expression:
              'await sobelFilter(input, { magnitude: true }) · await gaussianBlur(input, { sigma })',
            inputs: [
              { label: '输入矩阵 (3×3)', table: matrixToTable(sobelInputMatrix) },
            ],
            outputs: sobelBlurOutputs,
            highlight:
              'Sobel 返回水平/垂直梯度并可计算幅值；高斯模糊根据 σ 与核尺寸实现平滑。',
          },
          {
            title: '矩阵分解 (SVD / QR)',
            description: 'svd 与 qr 提供常见的分解形式，便于数值分析。',
            expression: 'svd(matrix) · qr(matrix)',
            inputs: [
              { label: 'SVD 输入 (3×3)', table: matrixToTable(svdInputMatrix) },
              { label: 'QR 输入 (3×3)', table: matrixToTable(qrInputMatrix) },
            ],
            outputs: [...svdOutputs, ...qrOutputs],
            highlight:
              'SVD 展示 U/Σ/Vᵀ 矩阵；QR 分解返回正交矩阵 Q 与上三角矩阵 R，可用于最小二乘等场景。',
          },
          {
            title: '线性方程与特征值',
            description: 'solve 求解 Ax = b，eigen 返回特征值与特征向量。',
            expression: 'solve(A, b) · eigen(A)',
            inputs: [
              { label: 'A (2×2)', table: matrixToTable(solveMatrixA) },
              { label: 'b (2×1)', table: matrixToTable(solveMatrixB) },
              { label: '特征分解矩阵 (2×2)', table: matrixToTable(eigenInputMatrix) },
            ],
            outputs: [...solveOutputs, ...eigenOutputs],
            highlight:
              'solve 返回线性方程组解向量；eigen 提供特征值数组和对应的特征向量矩阵。',
          },
        ],
      },
      {
        id: 'broadcast',
        heading: '广播与条件',
        summary:
          '广播机制让小尺寸张量扩展为目标形状，where 可用布尔掩码挑选元素。',
        demos: [
          {
            title: '行向量广播',
            description:
              'broadcastTo 将 1×3 向量复制成 3×3，再与原矩阵逐元素相加。',
            expression: 'add(base, broadcastTo(vector, 3, 3))',
            inputs: [
              { label: '基础矩阵 (3×3)', table: matrixToTable(baseGrid) },
              { label: '行向量 (1×3)', table: matrixToTable(rowVector) },
            ],
            outputs: [
              {
                label: '广播结果 (3×3)',
                table: matrixToTable(broadcastedRow),
              },
              {
                label: '加上偏移后的矩阵',
                table: matrixToTable(shiftedGrid),
              },
            ],
            highlight:
              'broadcastTo 复制行向量到目标行数，常用于为批量样本添加同一偏移。',
          },
          {
            title: '布尔掩码筛选',
            description:
              'where(mask, truthy, falsy) 按条件矩阵逐元素挑选数据。',
            expression: 'where(mask, data, zeros)',
            inputs: [
              { label: '广播矩阵', table: matrixToTable(shiftedGrid) },
              { label: '布尔掩码', table: matrixToTable(mask) },
            ],
            outputs: [
              { label: 'where 结果', table: matrixToTable(maskedGrid) },
            ],
            highlight:
              'falsy 分支提供默认值，这里与 0 组合，便于在可视化中突出命中的位置。',
          },
          {
            title: '压缩取值',
            description: 'compress(mask, matrix) 根据布尔掩码提取被选元素。',
            expression: 'compress(mask, matrix)',
            inputs: [
              { label: '矩阵', table: matrixToTable(shiftedGrid) },
              { label: '掩码', table: matrixToTable(mask) },
            ],
            outputs: [
              { label: '压缩结果 (列向量)', table: matrixToTable(compressedValues) },
            ],
            highlight:
              'compress 返回被掩码选中的元素（以列向量形式），适合快速筛选有效值或重构稀疏向量。',
          },
          {
            title: '列向量广播',
            description:
              'broadcastTo 也可以将 3×1 列向量扩展为 3×3，与原矩阵逐列相加。',
            expression: 'add(base, broadcastTo(column, 3, 3))',
            inputs: [
              { label: '基础矩阵 (3×3)', table: matrixToTable(baseGrid) },
              { label: '列向量 (3×1)', table: matrixToTable(columnVector) },
            ],
            outputs: [
              {
                label: '广播列向量 (3×3)',
                table: matrixToTable(broadcastedColumn),
              },
              {
                label: '列偏移后的矩阵',
                table: matrixToTable(columnShifted),
              },
            ],
            highlight:
              '这种列向量广播常用于按特征列做归一化或批量偏移。',
          },
        ],
      },
      {
        id: 'statistics',
        heading: '统计汇总',
        summary: '聚合函数可快速得到总和、均值与向量内积等指标。',
        demos: [
          {
            title: '聚合统计',
            description:
              'sum 会返回一个 1×1 矩阵，nanmean 自动忽略 NaN。',
            expression: 'sum(matrix), nanmean(matrix)',
            inputs: [
              { label: '观测矩阵', table: matrixToTable(statsMatrix) },
            ],
            outputs: [
              { label: 'sum(matrix)', scalar: totalSum },
              { label: 'nanmean(matrix)', scalar: meanValue },
            ],
            highlight:
              'WASM 后端使用补偿求和保证数值稳定，N-API 后端则委托原生实现。',
          },
          {
            title: '向量内积',
            description:
              'dot(a, b) 支持计算向量或矩阵的内积，返回 1×1 矩阵。',
            expression: 'dot(vectorA, vectorB)',
            inputs: [
              { label: '向量 A (1×3)', table: matrixToTable(dotVectorA) },
              { label: '向量 B (1×3)', table: matrixToTable(dotVectorB) },
            ],
            outputs: [{ label: 'dot(A, B)', scalar: dotValue }],
            highlight:
              'dot 要求形状一致，这里将两个向量都表示为 1×3。默认实现采用数值稳定的归约策略。',
          },
          {
            title: '更多聚合函数',
            description:
              'nansum、median、quantile、percentile 与 sumUnsafe 覆盖更多统计需求。',
            expression:
              'nansum(matrix), median(matrix), quantile(matrix, 0.75), percentile(matrix, 90), sumUnsafe(matrix)',
            inputs: [
              { label: '观测矩阵', table: matrixToTable(statsMatrix) },
            ],
            outputs: [
              { label: 'nansum(matrix)', scalar: nanSumValue },
              { label: 'median(matrix)', scalar: medianValue },
              { label: 'quantile(matrix, 0.75)', scalar: quantile75 },
              { label: 'percentile(matrix, 90)', scalar: percentile90 },
              { label: 'sumUnsafe(matrix)', scalar: sumUnsafeValue },
            ],
            highlight:
              'nansum 跳过 NaN；quantile/percentile 计算分位；sumUnsafe 提供无补偿求和结果便于性能对比。',
          },
        ],
      },
    ]
  }, [status])

const MatrixTable = ({ table }: { table: Table }) => (
  <table className="matrix-table">
    <tbody>
      {table.map((row, rIdx) => (
        <tr key={`row-${rIdx}`}>
          {row.map((value, cIdx) => (
            <td key={`cell-${rIdx}-${cIdx}`}>
              <code>{value}</code>
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>
)

function App() {
  const { status, backend, error } = useNumJSStatus()
  const linearAsyncData = useLinearAsyncData(status)
  const sections = useDemoSections(status, linearAsyncData)
  const [activeTab, setActiveTab] = useState<string | null>(null)

  const tabList = useMemo(() => {
    const base = sections.map((section) => ({
      id: section.id,
      label: section.heading,
      type: 'section' as const,
    }))

    return [
      ...base,
      {
        id: 'docs',
        label: '文档资源',
        type: 'docs' as const,
      },
    ]
  }, [sections])

  useEffect(() => {
    if (status !== 'ready') {
      setActiveTab(null)
      return
    }
    if (tabList.length === 0) {
      setActiveTab(null)
      return
    }
    setActiveTab((current) => {
      if (current && tabList.some((tab) => tab.id === current)) {
        return current
      }
      return tabList[0].id
    })
  }, [status, tabList])

  const activeSection = sections.find((section) => section.id === activeTab)

  return (
    <main className="app">
      <header className="hero">
        <h1>NumJS × Vite 示例</h1>
        <p>展示 @jayce789/numjs 在浏览器中的常见矩阵操作。</p>
        <div className={`status-pill status-${status}`}>
          {status === 'loading' && '正在加载 NumJS 后端…'}
          {status === 'ready' && `当前后端：${backend}`}
          {status === 'error' && `初始化失败：${error}`}
        </div>
      </header>

      {status === 'ready' ? (
        tabList.length > 0 && activeTab ? (
          <div className="tab-container">
            <nav className="tab-bar">
              {tabList.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`tab-button ${activeTab === tab.id ? 'is-active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </nav>

            <div className="tab-content">
              {activeSection ? (
                <section key={activeSection.id} className="demo-section">
                  <div className="section-header">
                    <h2>{activeSection.heading}</h2>
                    <p className="section-summary">{activeSection.summary}</p>
                  </div>
                  <div className="grid">
                    {activeSection.demos.map((demo) => (
                      <article key={demo.title} className="card">
                        <h3>{demo.title}</h3>
                        <p className="description">{demo.description}</p>
                        <div className="expression">
                          调用示例：<code>{demo.expression}</code>
                        </div>

                        <div className="matrices">
                          {demo.inputs.map((input) => (
                            <div key={input.label} className="matrix-block">
                              <span className="matrix-label">{input.label}</span>
                              <MatrixTable table={input.table} />
                            </div>
                          ))}
                        </div>

                        <div className="result">
                          <span className="matrix-label">输出</span>
                          <div className="outputs">
                            {demo.outputs.map((output) => (
                              <div key={output.label} className="output-block">
                                <span className="matrix-label">{output.label}</span>
                                {output.table ? (
                                  <MatrixTable table={output.table} />
                                ) : (
                                  <div className="scalar">
                                    <code>{output.scalar}</code>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>

                        {demo.highlight && (
                          <p className="highlight">{demo.highlight}</p>
                        )}
                      </article>
                    ))}
                  </div>
                </section>
              ) : activeTab === 'docs' ? (
                <section className="docs-section">
                  <div className="section-header">
                    <h2>文档入口</h2>
                    <p className="section-summary">
                      根据不同主题整理的 NumJS 文档与示例，便于继续深入探索。
                    </p>
                  </div>
                  <div className="doc-grid">
                    {documentationGroups.map((group) => (
                      <article key={group.title} className="card doc-card">
                        <h3>{group.title}</h3>
                        <p className="description">{group.description}</p>
                        <ul className="doc-links">
                          {group.links.map((link) => (
                            <li key={link.url}>
                              <a href={link.url} target="_blank" rel="noreferrer">
                                {link.label}
                              </a>
                              {link.note && (
                                <span className="doc-note">{link.note}</span>
                              )}
                            </li>
                          ))}
                        </ul>
                      </article>
                    ))}
                  </div>
                </section>
              ) : (
                <div className="empty-tab">请选择一个模块查看示例。</div>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-tab">暂无可用模块，请稍后重试。</div>
        )
      ) : (
        <section className="placeholder">
          {status === 'loading' && (
            <>
              <div className="spinner" />
              <p>正在准备 NumJS，请稍候…</p>
            </>
          )}
          {status === 'error' && (
            <>
              <p>初始化失败，请检查控制台日志。</p>
              <pre>{error}</pre>
            </>
          )}
        </section>
      )}

      <footer className="footer">
        <p>
          代码基于 Vite + React，适合直接扩展为更复杂的数值可视化页面。
        </p>
        <p>
          官方文档：{' '}
          <a
            href="https://github.com/jaycezhang789/numjs"
            target="_blank"
            rel="noreferrer"
          >
            @jayce789/numjs on GitHub
          </a>
        </p>
      </footer>
    </main>
  )
}

export default App
