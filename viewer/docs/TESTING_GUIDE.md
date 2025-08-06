# Testing Guide - Video Annotation Viewer

## 🎯 **Overview**

Video Annotation Viewer v0.2.0 now includes a comprehensive testing suite built with **Vitest** and **React Testing Library**. This guide shows you how to use the testing framework.

## 🚀 **Quick Start**

```bash
# Install dependencies (if not already done)
bun install

# Run tests in watch mode (recommended for development)
bun test

# Run all tests once and exit
bun run test:run

# Open interactive test UI in browser
bun run test:ui

# Generate coverage report
bun run test:coverage
```

## 📁 **Test Files Structure**

```
src/test/
├── setup.ts              # Test environment configuration
├── simple.test.ts         # Basic test examples ✅
├── fileUtils.test.ts      # File detection utilities
├── parsers.test.ts        # Data format parsers
├── components.test.tsx    # React component tests
├── integration.test.ts    # End-to-end workflows
├── debugUtils.test.ts     # Debug utility tests
└── README.md             # Detailed testing documentation
```

## 🧪 **What's Tested**

### ✅ **Core File System**
- File type detection (video, audio, JSON, WebVTT, RTTM)
- File size validation and constraints
- File set validation (video + annotation requirements)
- Human-readable type descriptions

### ✅ **Data Parsers**
- **WebVTT**: Speech recognition subtitle parsing
- **RTTM**: Speaker diarization format
- **Scene Detection**: JSON array processing
- **Error Handling**: Malformed data gracefully handled

### ✅ **React Components**
- **Footer**: Version info, external links, copyright
- **FileViewer**: Data inspection dialog
- Component rendering and user interactions

### ✅ **Integration Workflows**
- Complete drag & drop file detection pipeline
- COCO person tracking detection
- VideoAnnotator complete_results processing
- Face analysis file recognition
- Error scenarios and edge cases

## 📊 **Test Coverage**

Run coverage reports to see testing completeness:

```bash
bun run test:coverage
```

Opens `coverage/index.html` showing:
- **Line Coverage**: % of code lines executed
- **Branch Coverage**: % of code branches tested  
- **Function Coverage**: % of functions called
- **Statement Coverage**: % of statements executed

## 🔧 **Test Configuration**

### **vitest.config.ts**
- Configures Vite + React for testing
- Sets up jsdom environment (simulates browser)
- Configures path aliases (`@/` imports)
- Coverage reporting settings

### **src/test/setup.ts**
- Mocks browser APIs (matchMedia, ResizeObserver)
- Mocks video/canvas elements for media tests
- Mocks File/fetch APIs for upload tests
- Global test environment setup

## 📝 **Writing Tests**

### **Basic Test Example:**
```typescript
import { describe, it, expect } from 'vitest'

describe('My Feature', () => {
  it('should work correctly', () => {
    const result = myFunction('input')
    expect(result).toBe('expected')
  })
})
```

### **React Component Test:**
```typescript
import { render, screen, fireEvent } from '@testing-library/react'

it('should render and handle clicks', () => {
  render(<MyComponent />)
  
  fireEvent.click(screen.getByRole('button'))
  
  expect(screen.getByText('Success!')).toBeInTheDocument()
})
```

### **Async Test Example:**
```typescript
it('should handle file loading', async () => {
  const promise = loadFile('test.json')
  await expect(promise).resolves.toBeDefined()
})
```

## 🎨 **Test Categories**

### **Unit Tests** (`*.test.ts`)
- Test individual functions in isolation
- Fast execution, focused scope
- Examples: `fileUtils.test.ts`, `parsers.test.ts`

### **Component Tests** (`*.test.tsx`)
- Test React component rendering and behavior
- User interaction simulation
- Examples: `components.test.tsx`

### **Integration Tests** (`integration.test.ts`)
- Test complete workflows end-to-end
- Multiple components working together
- File detection pipeline testing

## 🚨 **Common Testing Patterns**

### **File Upload Testing**
```typescript
const file = new File(['content'], 'test.json', { type: 'application/json' })
const result = await detectFileType(file)
expect(result.type).toBe('person_tracking')
```

### **Component Interaction Testing**
```typescript
render(<FileUploader onLoad={mockCallback} />)
const input = screen.getByLabelText(/upload/i)
fireEvent.change(input, { target: { files: [mockFile] } })
expect(mockCallback).toHaveBeenCalled()
```

### **Error Handling Testing**
```typescript
it('should handle malformed JSON', async () => {
  const badFile = new File(['{ invalid json'], 'bad.json')
  const result = await parseJSON(badFile)
  expect(result.type).toBe('unknown')
})
```

## 🔍 **Best Practices**

1. **Descriptive Names**: `should detect COCO files correctly` ✅ vs `test1` ❌
2. **Arrange-Act-Assert**: Set up → Execute → Verify
3. **Test Behavior**: What it does, not how it does it
4. **Mock Dependencies**: Don't make real API calls in tests
5. **Edge Cases**: Empty inputs, errors, boundary conditions

## 📈 **Testing Workflow**

### **Development Cycle:**
1. **Write failing test** → Red ❌
2. **Write minimal code** → Green ✅  
3. **Refactor & improve** → Blue 🔵
4. **Repeat for next feature**

### **Before Commits:**
```bash
bun run test:run    # Ensure all tests pass
bun run lint        # Check code style
bun run build       # Verify build works
```

## 🎯 **Testing Philosophy**

The testing suite ensures:
- **Reliability**: Catch bugs before users do
- **Confidence**: Safe refactoring and feature addition
- **Documentation**: Tests show how code should work
- **Quality**: Maintain high standards as codebase grows

## 📚 **Resources**

- **[Vitest Documentation](https://vitest.dev/)** - Test runner
- **[React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)** - Component testing
- **[Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)** - Kent C. Dodds guide

---

## 🎉 **Testing Success!**

With this testing framework, Video Annotation Viewer v0.2.0 is now enterprise-ready with:
- **44 test cases** covering critical functionality
- **Unit, integration, and component test coverage** 
- **Automated regression prevention**
- **Developer-friendly test workflow**

Happy testing! 🧪✨