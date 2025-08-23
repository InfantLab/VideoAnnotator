import { Outlet } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link, useLocation } from "react-router-dom";
import { Plus, Database, List, Settings } from "lucide-react";

const CreateLayout = () => {
  const location = useLocation();

  const navigationItems = [
    { path: "/create/jobs", label: "Jobs", icon: List },
    { path: "/create/new", label: "New Job", icon: Plus },
    { path: "/create/datasets", label: "Datasets", icon: Database },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            VideoAnnotator Control Panel
          </h1>
          <p className="text-gray-600">
            Create and manage annotation jobs using the VideoAnnotator pipeline
          </p>
        </div>

        {/* Navigation */}
        <div className="mb-6">
          <nav className="flex space-x-2">
            {navigationItems.map(({ path, label, icon: Icon }) => (
              <Link key={path} to={path}>
                <Button
                  variant={location.pathname === path ? "default" : "outline"}
                  className="flex items-center gap-2"
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Button>
              </Link>
            ))}
          </nav>
        </div>

        {/* Content */}
        <Card className="min-h-[600px] p-6">
          <Outlet />
        </Card>
      </div>
    </div>
  );
};

export default CreateLayout;