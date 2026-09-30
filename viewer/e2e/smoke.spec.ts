import { test, expect } from '@playwright/test'

test('app loads and shows title', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/Video Annotation Viewer/i)
    await expect(page.locator('h1')).toContainText(/Video Annotation Viewer/i)
})
