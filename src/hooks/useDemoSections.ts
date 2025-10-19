import { useMemo } from 'react'
import {
  Matrix,
  add,
  broadcastTo,
  div,
  dot,
  im2col,
  matrixFromFixed,
  matmul,
  maxPool,
  median,
  mul,
  nanmean,
  nansum,
  neg,
  quantile,
  percentile,
  compress,
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
import type { DemoOutput, DemoSection, LinearAsyncResult, Status } from '../types'
import {
  EIGEN_INPUT_VALUES,
  MATMUL_LEFT_VALUES,
  MATMUL_RIGHT_VALUES,
  CONV_INPUT_VALUES,
  CONV_KERNEL_VALUES,
  POOL_INPUT_VALUES,
  SOBEL_INPUT_VALUES,
  SVD_INPUT_VALUES,
  QR_INPUT_VALUES,
  SOLVE_MATRIX_VALUES,
  SOLVE_B_VALUES,
} from '../data/demoConstants'
import {
  truncate,
  formatNumber,
  formatFloatArray,
  matrixToTable,
} from '../utils/matrixHelpers'
import {
  safeGather,
  safeGatherPairs,
  safePut,
  safeScatter,
  safeScatterPairs,
  safeTake,
} from '../utils/safeMatrixOps'

export const useDemoSections = (status: Status, asyncData: LinearAsyncResult) =>
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
  }, [status, asyncData])
