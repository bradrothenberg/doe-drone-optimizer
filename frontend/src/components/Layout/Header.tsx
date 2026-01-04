import { Box, Typography } from '@mui/material'

export default function Header() {
  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        color: '#000000',
        p: 3,
        borderBottom: '2px solid #cccccc',
        textAlign: 'center'
      }}
    >
      <Typography
        variant="h3"
        sx={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontWeight: 600,
          fontSize: '2em',
          mb: 0.5,
          letterSpacing: '0.5px'
        }}
      >
        DOE Drone Design Optimizer
      </Typography>
      <Typography
        variant="h5"
        sx={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontWeight: 400,
          fontSize: '1.1em',
          mt: 0.5,
          letterSpacing: '0.3px',
          opacity: 0.8
        }}
      >
        Multi-Objective Pareto Optimization
      </Typography>
      <Typography
        variant="body2"
        sx={{
          fontSize: '0.9em',
          opacity: 0.7,
          letterSpacing: '1px',
          mt: 1
        }}
      >
        NSGA-II · Ensemble ML · Interactive Exploration
      </Typography>
    </Box>
  )
}
