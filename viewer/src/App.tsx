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
import Index from "./pages/Index";
import Home from "./pages/Home";
import GettingStarted from "./pages/GettingStarted";
import NotFound from "./pages/NotFound";
import Jobs from "./pages/Jobs";
import BatchDetail from "./pages/BatchDetail";
import JobDetail from "./pages/JobDetail";
import NewJob from "./pages/NewJob";
import Settings from "./pages/Settings";
import JobResultsViewer from "./pages/JobResultsViewer";
import Library from "./pages/Library";

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
                  <Route path="/viewer" element={<Index />} />
                  <Route path="/view/:jobId" element={<JobResultsViewer />} />

                  {/* Routes with shared AppLayout navigation */}
                  <Route element={<AppLayout />}>
                    <Route path="/" element={<Home />} />
                    <Route path="/getting-started" element={<GettingStarted />} />
                    <Route path="/library" element={<Library />} />
                    <Route path="/jobs" element={<Jobs />} />
                    <Route path="/jobs/:jobId" element={<JobDetail />} />
                    <Route path="/jobs/new" element={<NewJob />} />
                    {/* Runs (batches) are listed on the Jobs page. */}
                    <Route path="/batches" element={<Navigate to="/jobs" replace />} />
                    <Route path="/batches/:batchId" element={<BatchDetail />} />
                    {/* Was a "Coming Soon" placeholder; datasets the viewer has are in the Library. */}
                    <Route path="/datasets" element={<Navigate to="/library" replace />} />
                    <Route path="/settings" element={<Settings />} />
                  </Route>

                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={<NotFound />} />
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
