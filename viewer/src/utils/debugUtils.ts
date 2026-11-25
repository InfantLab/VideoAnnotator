/**
 * Debug utilities for the browser console: the demo files and the API.
 */

import { apiClient } from '../api/client'
import { STANDALONE_API_URL } from '../lib/apiConnection'
import { DEMO_DATA_SETS, loadDemoFiles, type DemoKey } from '../lib/demos'

/** Whether every file of a demo can be fetched. */
export async function checkDataIntegrity(key: DemoKey): Promise<{ valid: boolean; issues: string[] }> {
  const { video, results } = DEMO_DATA_SETS[key]
  const issues: string[] = []
  for (const url of [video, ...results]) {
    try {
      const response = await fetch(url, { method: 'HEAD' })
      if (!response.ok) issues.push(`${url}: ${response.status}`)
    } catch (error) {
      issues.push(`${url}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  return { valid: issues.length === 0, issues }
}

// API Testing utilities (from browser_debug_console.js)
const createVideoAnnotatorDebug = () => {
  // Use the same configuration as the actual API client for consistency
  const getApiConfig = () => {
    if (typeof window !== 'undefined') {
      return apiClient.getConfig()
    }
    return {
      baseURL: STANDALONE_API_URL,
      token: ''
    }
  }

  return {
    get apiBase() { return getApiConfig().baseURL },
    get defaultToken() { return getApiConfig().token },
    logRequests: true,

    async checkHealth() {
      console.log('🏥 Checking API health...')
      try {
        const response = await fetch(`${this.apiBase}/health`)
        const data = await response.json()

        console.log('✅ Basic Health:', response.status === 200 ? 'OK' : 'FAIL')
        console.table({
          'Status': data.status,
          'API Version': data.api_version,
          'Server': data.videoannotator_version
        })

        return data
      } catch (error) {
        console.error('❌ Health check failed:', error)
        return null
      }
    },

    async checkAuth(token = null) {
      const authToken = token || this.defaultToken
      console.log(`🔐 Testing authentication with token: ${authToken.substring(0, 10)}...`)

      try {
        const response = await fetch(`${this.apiBase}/api/v1/debug/token-info`, {
          headers: { 'Authorization': `Bearer ${authToken}` }
        })

        if (response.ok) {
          const data = await response.json()
          console.log('✅ Authentication: Valid')
          console.table({
            'User ID': data.token?.user_id,
            'Valid': data.token?.valid,
            'Permissions': data.token?.permissions?.join(', '),
            'Rate Limit': `${data.token?.rate_limit?.remaining_this_minute || '?'}/min remaining`
          })
          return data
        } else {
          console.error('❌ Authentication failed:', response.status, await response.text())
          return null
        }
      } catch (error) {
        console.error('❌ Auth check failed:', error)
        return null
      }
    },

    async runAllTests(token = null) {
      console.log('🧪 Running comprehensive API tests...')
      console.log('='.repeat(50))

      const results = {
        health: await this.checkHealth(),
        auth: await this.checkAuth(token)
      }

      console.log('='.repeat(50))
      console.log('📊 Test Results Summary:')
      Object.entries(results).forEach(([test, result]) => {
        console.log(`- ${test}: ${result ? '✅' : '❌'}`)
      })

      return results
    }
  }
}

// Make utilities available globally for browser console testing
if (typeof window !== 'undefined') {
  const windowWithDebug = window as unknown as {
    debugUtils?: unknown;
    VideoAnnotatorDebug?: unknown;
  };

  windowWithDebug.debugUtils = { DEMO_DATA_SETS, loadDemoFiles, checkDataIntegrity }
  windowWithDebug.VideoAnnotatorDebug = createVideoAnnotatorDebug()
}
