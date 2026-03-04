import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
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
import JobDetail from "./pages/JobDetail";
import NewJob from "./pages/NewJob";
import Datasets from "./pages/Datasets";
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
              <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
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
                    <Route path="/datasets" element={<Datasets />} />
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
