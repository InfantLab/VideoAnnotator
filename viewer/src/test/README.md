# Testing Guide for Video Annotation Viewer

This directory contains comprehensive unit and integration tests for the Video Annotation Viewer.

## 🧪 **Test Framework: Vitest + React Testing Library**

- **[Vitest](https://vitest.dev/)**: Fast unit test runner built for Vite projects
- **[React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)**: Testing utilities for React components
- **[jsdom](https://github.com/jsdom/jsdom)**: DOM implementation for Node.js (simulates browser environment)

## 📋 **Available Test Commands**

```bash
# Run tests in watch mode (recommended for development)
bun test

# Run tests once and exit
bun run test:run

# Run tests with UI (browser-based test interface)
bun run test:ui

# Run tests with coverage report
bun run test:coverage
```

## 📁 **Test Structure**

```
src/test/
├── setup.ts              # Test environment setup and mocks
├── fileUtils.test.ts      # File detection and validation utilities
├── parsers.test.ts        # WebVTT, RTTM, Scene detection parsers  
├── components.test.tsx    # React component tests (Footer, FileViewer)
├── integration.test.ts    # End-to-end file detection workflow
├── debugUtils.test.ts     # Demo loading and debug utilities
└── README.md             # This guide
```

## 🎯 **What's Tested**

### ✅ **Core Utilities** (`fileUtils.test.ts`)
- File type detection (video, audio, JSON, WebVTT, RTTM)
- File size validation 
- File set validation (requires video + annotation files)
- Human-readable descriptions for file types

### ✅ **Data Parsers** (`parsers.test.ts`)
- **WebVTT Parser**: Speech recognition subtitle parsing
- **RTTM Parser**: Speaker diarization format parsing
- **Scene Parser**: Scene detection JSON array parsing
- Error handling for malformed data

### ✅ **React Components** (`components.test.tsx`) 
- **Footer**: Version info, GitHub/VideoAnnotator links, copyright
- **FileViewer**: Annotation data display dialog
- Component rendering and user interactions

### ✅ **Integration Workflows** (`integration.test.ts`)
- Complete drag & drop file detection pipeline
- COCO person tracking detection
- VideoAnnotator complete_results detection
- Face analysis file detection
- Error handling for malformed/unknown files

### ✅ **Debug Utilities** (`debugUtils.test.ts`)
- Demo dataset configuration validation
- File loading workflow testing
- Data integrity checking

## 🔧 **Test Environment Setup**

The `setup.ts` file configures the test environment with:

- **DOM Mocks**: `window.matchMedia`, `ResizeObserver`
- **Media Mocks**: `HTMLVideoElement.play()`, `HTMLVideoElement.pause()`
- **Canvas Mocks**: `HTMLCanvasElement.getContext()` for overlay rendering tests
- **File System Mocks**: `fetch()`, `File` constructor for file upload tests

## 🚀 **Running Your First Tests**

1. **Install dependencies:**
   ```bash
   bun install
   ```

2. **Run tests in watch mode:**
   ```bash
   bun test
   ```

3. **View test results in your terminal**

4. **Optional: Open the UI for interactive testing:**
   ```bash
   bun run test:ui
   ```
   Then open http://localhost:51204 in your browser

## 📊 **Coverage Reports**

Generate test coverage reports to see what code is tested:

```bash
bun run test:coverage
```

This creates a `coverage/` directory with HTML reports showing:
- Line coverage percentages
- Uncovered code branches  
- Visual coverage maps

## 🐛 **Writing New Tests**

### Example: Testing a Utility Function
```typescript
import { describe, it, expect } from 'vitest'
import { myFunction } from '../lib/myModule'

describe('myFunction', () => {
  it('should do something specific', () => {
    const result = myFunction('input')
    expect(result).toBe('expected output')
  })
})
```

### Example: Testing a React Component
```typescript
import { render, screen, fireEvent } from '@testing-library/react'
import { MyComponent } from '../components/MyComponent'

it('should render and handle click', () => {
  render(<MyComponent />)
  
  const button = screen.getByRole('button')
  fireEvent.click(button)
  
  expect(screen.getByText('Clicked!')).toBeInTheDocument()
})
```

## 🔍 **Best Practices**

1. **Test behavior, not implementation** - Focus on what the code does, not how
2. **Use descriptive test names** - `should detect COCO files correctly` vs `test1`
3. **Arrange-Act-Assert** - Set up data, perform action, check result
4. **Mock external dependencies** - Don't make real network requests in tests
5. **Test edge cases** - Empty inputs, malformed data, error conditions

## 🚨 **Common Issues & Solutions**

**Tests fail with "ReferenceError: File is not defined"**
→ The File constructor is mocked in `setup.ts`, make sure it's imported

**Canvas/Video tests fail**  
→ DOM APIs are mocked in `setup.ts` for the test environment

**Import errors with `@/` paths**
→ Path aliases are configured in `vitest.config.ts`

**Async tests timeout**
→ Use `await` for promises and increase timeout if needed

## 📚 **Resources**

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library Cheat Sheet](https://testing-library.com/docs/react-testing-library/cheatsheet)
- [Jest DOM Matchers](https://github.com/testing-library/jest-dom) (used by @testing-library/jest-dom)

Happy testing! 🧪✨