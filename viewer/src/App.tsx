import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SSEProvider } from "@/contexts/SSEContext";
import { PipelineProvider } from "@/contexts/PipelineContext";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Create from "./pages/Create";
import CreateJobs from "./pages/CreateJobs";
import CreateJobDetail from "./pages/CreateJobDetail";
import CreateNewJob from "./pages/CreateNewJob";
import CreateDatasets from "./pages/CreateDatasets";
import CreateSettings from "./pages/CreateSettings";

const queryClient = new QueryClient();

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <SSEProvider enabled={false}>
        <PipelineProvider>
          <TooltipProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/create" element={<Create />}>
                  <Route path="jobs" element={<CreateJobs />} />
                  <Route path="jobs/:jobId" element={<CreateJobDetail />} />
                  <Route path="new" element={<CreateNewJob />} />
                  <Route path="datasets" element={<CreateDatasets />} />
                  <Route path="settings" element={<CreateSettings />} />
                  <Route index element={<CreateJobs />} />
                </Route>
                {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </BrowserRouter>
          </TooltipProvider>
        </PipelineProvider>
      </SSEProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
