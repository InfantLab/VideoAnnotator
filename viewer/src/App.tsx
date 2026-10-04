import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { SSEProvider } from "@/contexts/SSEContext";
import { PipelineProvider } from "@/contexts/PipelineProvider";
import { ServerCapabilitiesProvider } from "@/contexts/ServerCapabilitiesProvider";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { AppLayout } from "@/components/AppLayout";
import { lazy, Suspense, type ReactNode } from "react";
import Home from "./pages/Home";

// Every page but Home loads on first visit, keeping the initial bundle within
// the constitution's 300 KB gzipped (it was one 300+ KB chunk).
const Index = lazy(() => import("./pages/Index"));
const GettingStarted = lazy(() => import("./pages/GettingStarted"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Jobs = lazy(() => import("./pages/Jobs"));
const BatchDetail = lazy(() => import("./pages/BatchDetail"));
const JobDetail = lazy(() => import("./pages/JobDetail"));
const NewJob = lazy(() => import("./pages/NewJob"));
const Settings = lazy(() => import("./pages/Settings"));
const JobResultsViewer = lazy(() => import("./pages/JobResultsViewer"));
const Library = lazy(() => import("./pages/Library"));
const Datasets = lazy(() => import("./pages/Datasets"));
const Prompts = lazy(() => import("./pages/Prompts"));
const Workbench = lazy(() => import("./pages/Workbench"));
const Compare = lazy(() => import("./pages/Compare"));

/** Per page, so the navigation stays on screen while a page loads. */
const page = (element: ReactNode) => (
  <Suspense fallback={<div className="p-8 text-sm text-muted-foreground">Loading…</div>}>{element}</Suspense>
);

const queryClient = new QueryClient();

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <ServerCapabilitiesProvider>
        <SSEProvider enabled={false}>
          <PipelineProvider>
            <TooltipProvider>
              <Toaster />
              <Sonner />
              <BrowserRouter
                basename={import.meta.env.BASE_URL}
                future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
              >
                <Routes>
                  {/* Full-screen routes (no shared nav) */}
                  <Route path="/viewer" element={page(<Index />)} />
                  <Route path="/view/:jobId" element={page(<JobResultsViewer />)} />

                  {/* Routes with shared AppLayout navigation */}
                  <Route element={<AppLayout />}>
                    <Route path="/" element={<Home />} />
                    <Route path="/getting-started" element={page(<GettingStarted />)} />
                    <Route path="/library" element={page(<Library />)} />
                    <Route path="/jobs" element={page(<Jobs />)} />
                    <Route path="/jobs/:jobId" element={page(<JobDetail />)} />
                    <Route path="/jobs/new" element={page(<NewJob />)} />
                    {/* Runs (batches) are listed on the Jobs page. */}
                    <Route path="/batches" element={<Navigate to="/jobs" replace />} />
                    <Route path="/batches/:batchId" element={page(<BatchDetail />)} />
                    <Route path="/datasets" element={page(<Datasets />)} />
                    <Route path="/prompts" element={page(<Prompts />)} />
                    <Route path="/workbench" element={page(<Workbench />)} />
                    <Route path="/compare" element={page(<Compare />)} />
                    <Route path="/settings" element={page(<Settings />)} />
                  </Route>

                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={page(<NotFound />)} />
                </Routes>
              </BrowserRouter>
            </TooltipProvider>
          </PipelineProvider>
        </SSEProvider>
      </ServerCapabilitiesProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
