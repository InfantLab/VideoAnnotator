import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock window.matchMedia (used by many UI components)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock window.ResizeObserver (used by some UI components)
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock HTMLVideoElement methods (for video player tests)
Object.defineProperty(HTMLMediaElement.prototype, 'play', {
  writable: true,
  value: vi.fn().mockImplementation(() => Promise.resolve()),
})

Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
  writable: true,
  value: vi.fn(),
})

Object.defineProperty(HTMLMediaElement.prototype, 'load', {
  writable: true,
  value: vi.fn(),
})

// Mock canvas context (for overlay rendering tests)
HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((contextId) => {
  if (contextId === '2d') {
    return {
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      arc: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 100 })),
      drawImage: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      translate: vi.fn(),
      rotate: vi.fn(),
      scale: vi.fn(),
      setTransform: vi.fn(),
      getImageData: vi.fn(),
      putImageData: vi.fn(),
      createImageData: vi.fn(),
      globalAlpha: 1,
      globalCompositeOperation: 'source-over',
      strokeStyle: '#000000',
      fillStyle: '#000000',
      lineWidth: 1,
      lineCap: 'butt',
      lineJoin: 'miter',
      miterLimit: 10,
      font: '10px sans-serif',
      textAlign: 'start',
      textBaseline: 'alphabetic',
    }
  }
  return null
})

// Mock fetch for file loading tests
global.fetch = vi.fn()

// Mock File constructor for file upload tests
global.File = class MockFile {
  constructor(public chunks: any[], public name: string, public options: any = {}) {}
  get size() { return this.chunks.reduce((size, chunk) => size + chunk.length, 0) }
  get type() { return this.options.type || '' }
  text() { return Promise.resolve(this.chunks.join('')) }
  slice(start?: number, end?: number) { 
    const content = this.chunks.join('').slice(start, end)
    return { text: () => Promise.resolve(content) }
  }
} as any