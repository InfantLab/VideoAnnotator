import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { SSEProvider } from "@/contexts/SSEContext";
import { PipelineProvider } from "@/contexts/PipelineProvider";
import { ServerCapabilitiesProvider } from "@/contexts/ServerCapabilitiesProvider";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { AppLayout } from "@/components/AppLayout";
import { lazy, Suspense, type ReactNode } from "react";
import Home from "./pages/Home";
import { AskRedirect } from "./components/ask/AskRedirect";
import { NotifyNavigator } from "./components/NotifyNavigator";
import { RootRoute } from "./components/RootRoute";
import { BatchRedirect, JobRedirect, RedirectWithState } from "./components/Redirects";

// Every page but Home loads on first visit, keeping the initial bundle within
// the constitution's 300 KB gzipped (it was one 300+ KB chunk).
const Index = lazy(() => import("./pages/Index"));
const Help = lazy(() => import("./pages/Help"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Settings = lazy(() => import("./pages/Settings"));
const VideoPage = lazy(() => import("./pages/VideoPage"));
const Compare = lazy(() => import("./pages/Compare"));
const AskAI = lazy(() => import("./pages/AskAI"));
const Pipelines = lazy(() => import("./pages/Pipelines"));
const Videos = lazy(() => import("./pages/Videos"));
const NewRun = lazy(() => import("./pages/NewRun"));

/** Per page, so the navigation stays on screen while a page loads. */
const page = (element: ReactNode) => (
  <Suspense fallback={<div className="p-8 text-sm text-muted-foreground">Loading…</div>}>{element}</Suspense>
);


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
                <NotifyNavigator />
                <Routes>
                  {/* Full-screen routes (no shared nav) */}
                  <Route path="/viewer" element={page(<Index />)} />
                  <Route path="/view/:jobId" element={<JobRedirect />} />
                  <Route path="/video" element={page(<VideoPage />)} />

                  {/* Routes with shared AppLayout navigation */}
                  <Route element={<AppLayout />}>
                    <Route path="/" element={<RootRoute />} />
                    <Route path="/home" element={<Home />} />
                    <Route path="/videos" element={page(<Videos />)} />
                    <Route path="/run" element={page(<NewRun />)} />
                    <Route path="/help" element={page(<Help />)} />
                    {/* Spec 029 R13: every old address leads somewhere. */}
                    <Route path="/getting-started" element={<Navigate to="/help" replace />} />
                    <Route path="/results" element={<Navigate to="/videos" replace />} />
                    <Route path="/library" element={<Navigate to="/videos" replace />} />
                    <Route path="/jobs" element={<Navigate to="/videos" replace />} />
                    <Route path="/jobs/new" element={<RedirectWithState to="/run" />} />
                    <Route path="/jobs/:jobId" element={<JobRedirect />} />
                    <Route path="/batches" element={<Navigate to="/videos" replace />} />
                    <Route path="/batches/:batchId" element={<BatchRedirect />} />
                    <Route path="/datasets" element={<Navigate to="/videos" replace />} />
                    {/* Ask AI replaced the Prompts page and the Workbench (spec 027); old links still land. */}
                    <Route path="/prompts" element={<AskRedirect history />} />
                    <Route path="/pipelines" element={page(<Pipelines />)} />
                    <Route path="/workbench" element={<AskRedirect />} />
                    <Route path="/compare" element={page(<Compare />)} />
                    <Route path="/ask" element={page(<AskAI />)} />
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
