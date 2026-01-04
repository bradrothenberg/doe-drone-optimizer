import axios from 'axios'
import type { OptimizeRequest, OptimizeResponse, PredictRequest, PredictResponse } from '../types'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json'
  }
})

export const optimizeDesigns = async (request: OptimizeRequest): Promise<OptimizeResponse> => {
  const response = await api.post<OptimizeResponse>('/optimize', request)
  return response.data
}

export const predictDesigns = async (request: PredictRequest): Promise<PredictResponse> => {
  const response = await api.post<PredictResponse>('/predict', request)
  return response.data
}

export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}
