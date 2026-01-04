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
  displayModeBar: true,
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: [
    'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d',
    'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'
  ] as const,
  toImageButtonOptions: {
    format: 'png' as const,
    filename: 'pareto_chart',
    height: 600,
    width: 800,
    scale: 2
  }
}

export const chart3DConfig = {
  displayModeBar: true,
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: [
    'hoverClosest3d'
  ] as const,
  toImageButtonOptions: {
    format: 'png' as const,
    filename: 'pareto_3d_chart',
    height: 600,
    width: 800,
    scale: 2
  }
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
