/**
 * Shared Plotly chart configurations matching DOE report
 */

export const chartLayout = {
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  font: { family: 'Courier New, monospace', color: '#000000', size: 11 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  hovermode: 'closest' as const,
  xaxis: {
    gridcolor: '#e0e0e0',
    linecolor: '#cccccc',
    tickfont: { size: 10 }
  },
  yaxis: {
    gridcolor: '#e0e0e0',
    linecolor: '#cccccc',
    tickfont: { size: 10 }
  }
}

export const chartConfig = {
  displayModeBar: false,
  responsive: true
}

export const paretoMarker = {
  size: 10,
  symbol: 'star' as const,
  color: '#d32f2f',
  line: { color: '#b71c1c', width: 1 }
}

export const paretoLine = {
  color: '#d32f2f',
  width: 2,
  dash: 'dot' as const
}

export const dominatedMarker = {
  size: 4,
  color: '#cccccc',
  line: { color: '#999999', width: 0.5 }
}
