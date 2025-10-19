import type { Matrix } from '@jayce789/numjs'

export type Status = 'loading' | 'ready' | 'error'

export type Table = string[][]

export type DemoMatrix = {
  label: string
  table: Table
}

export type DemoOutput = {
  label: string
  table?: Table
  scalar?: string
}

export type DemoBlock = {
  title: string
  description: string
  expression: string
  inputs: DemoMatrix[]
  outputs: DemoOutput[]
  highlight?: string
}

export type DemoSection = {
  id: string
  heading: string
  summary: string
  demos: DemoBlock[]
}

export type LinearAsyncResult = {
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

export type DocLink = {
  label: string
  url: string
  note?: string
}

export type DocGroup = {
  title: string
  description: string
  links: DocLink[]
}
