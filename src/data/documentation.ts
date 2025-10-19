export const documentationGroups = [
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
