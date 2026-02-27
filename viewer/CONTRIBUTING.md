# Contributing to Video Annotation Viewer

Thank you for your interest in contributing to Video Annotation Viewer! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/video-annotation-viewer.git
   cd video-annotation-viewer
   ```
3. Install dependencies:
   ```bash
   bun install
   # or npm install
   ```
4. Start the development server:
   ```bash
   bun run dev
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run linting and tests before committing:
   ```bash
   bun run lint
   bun run test:run
   ```
4. Commit with a clear message describing the change
5. Push to your fork and open a Pull Request against `main`

## Code Standards

- **Language**: TypeScript with strict mode
- **Framework**: React 18 with functional components and hooks
- **Styling**: Tailwind CSS with shadcn/ui components
- **Testing**: Vitest for unit/integration tests, Playwright for end-to-end tests
- **Linting**: ESLint — run `bun run lint` before submitting

## Running Tests

```bash
# Unit and integration tests
bun run test:run

# Tests with coverage report
bun run test:coverage

# End-to-end tests (requires Playwright browsers)
bun run e2e:install   # first time only
bun run e2e
```

## Reporting Issues

- Use [GitHub Issues](https://github.com/InfantLab/video-annotation-viewer/issues) to report bugs or request features
- Include steps to reproduce, expected behavior, and actual behavior
- Attach screenshots or error messages where relevant

## Getting Help

- Email: infantologist@gmail.com
- See the [Developer Guide](docs/DEVELOPER_GUIDE.md) for architecture details
- See the [File Formats Guide](docs/FILE_FORMATS.md) for supported data formats

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.
