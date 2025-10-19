# NumJS 前端示例

演示如何使用 [@jayce789/numjs](https://github.com/jaycezhang789/numjs) 在浏览器端运行常见矩阵与统计计算。

## 功能概览

- **矩阵基础**：逐元素运算、拼接、堆叠、索引/写入与序列化示例。  
- **线性代数**：矩阵乘法、转置、行列视图与切片。  
- **广播与条件**：行/列向量广播与 `where` 条件筛选。  
- **统计汇总**：`sum`/`nanmean`/`nansum`/`median`/`percentile` 等聚合函数，以及向量内积。
- **文档入口**：按照模块整理的 NumJS 官方文档链接。

## 快速开始

```bash
npm install
npm run dev
```

随后访问终端输出的本地地址（默认 <http://localhost:5173/>），使用顶部标签切换不同模块的演示界面。

## 构建发布版本

```bash
npm run build
```

构建结果位于 `dist/`，可通过 `npm run preview` 进行本地预览。

## 目录结构

```
├── public/            # 静态资源
├── src/
│   ├── App.tsx        # 主界面，包含所有演示模块
│   ├── App.css        # 组件样式（已针对宽屏布局优化）
│   ├── index.css      # 全局基础样式
│   └── main.tsx       # React 入口
├── vite.config.ts     # Vite + React 配置
└── package.json
```

## 兼容性说明

- 项目使用 Vite + React + TypeScript 开发，需 Node.js ≥ 18。  
- NumJS 会在浏览器端自动加载 WebAssembly 后端，部分 API（如稀疏索引写入）在 WASM 环境暂未实现，示例代码提供了 JS 回退逻辑，确保演示功能稳定。

## 参考资料

- [NumJS 仓库](https://github.com/jaycezhang789/numjs)  
- [NumJS WebGPU 教程](https://github.com/jaycezhang789/numjs/blob/main/docs/tutorials/webgpu.md)  
- [从 NumPy 迁移指南](https://github.com/jaycezhang789/numjs/blob/main/docs/tutorials/from-numpy-migration.md)
