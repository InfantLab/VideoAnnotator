import { describe, it, expect } from 'vitest'

describe('Simple Test Setup', () => {
  it('should run basic assertions', () => {
    expect(1 + 1).toBe(2)
    expect('hello').toContain('ell')
    expect([1, 2, 3]).toHaveLength(3)
  })

  it('should work with async tests', async () => {
    const promise = Promise.resolve('success')
    await expect(promise).resolves.toBe('success')
  })

  it('should handle objects', () => {
    const obj = { name: 'Video Annotation Viewer', version: '0.2.0' }
    expect(obj).toHaveProperty('name')
    expect(obj).toEqual({
      name: 'Video Annotation Viewer',
      version: '0.2.0'
    })
  })
})