import type { Table } from '../types'

type MatrixTableProps = {
  table: Table
}

export const MatrixTable = ({ table }: MatrixTableProps) => (
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
