/**
 * Color constants extracted from DOE report
 */

export const colors = {
  // Background & text
  background: '#ffffff',
  text: '#000000',
  textSecondary: '#666666',
  cardBackground: '#f5f5f5',
  border: '#cccccc',
  borderLight: '#e0e0e0',
  gridColor: '#e0e0e0',

  // Pareto markers
  paretoOptimal: '#d32f2f',
  paretoOptimalBorder: '#b71c1c',
  dominated: '#cccccc',
  dominatedBorder: '#999999',

  // Color scales
  rangeColorscale: [
    [0, '#ffcccc'], // Low - light red
    [0.5, '#ffffcc'], // Mid - light yellow
    [1, '#c8e6c9']  // High - light green
  ],

  mtowColorscale: [
    [0, '#e3f2fd'], // Low - light blue
    [1, '#1565c0']  // High - dark blue
  ],

  // Individual colors for continuous mapping
  range: {
    low: '#ffcccc',
    mid: '#ffffcc',
    high: '#c8e6c9'
  },

  mtow: {
    low: '#e3f2fd',
    high: '#1565c0'
  }
}

export const fonts = {
  primary: "'Courier New', 'Consolas', 'Monaco', monospace",
  header: "'IBM Plex Mono', monospace"
}
