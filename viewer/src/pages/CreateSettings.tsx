import { TokenSetup } from '@/components/TokenSetup';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Settings, Key, Server, HelpCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import vavIcon from '@/assets/v-a-v.icon.png';

const CreateSettings = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
        <div className="flex items-center gap-2">
          <Settings className="h-6 w-6" />
          <h2 className="text-2xl font-semibold">Settings</h2>
        </div>
      </div>

      <Tabs defaultValue="api" className="space-y-4">
        <TabsList>
          <TabsTrigger value="api" className="flex items-center gap-2">
            <Key className="h-4 w-4" />
            API Configuration
          </TabsTrigger>
          <TabsTrigger value="server" className="flex items-center gap-2">
            <Server className="h-4 w-4" />
            Server Info
          </TabsTrigger>
          <TabsTrigger value="help" className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            Help
          </TabsTrigger>
        </TabsList>

        <TabsContent value="api">
          <TokenSetup />
        </TabsContent>

        <TabsContent value="server" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Server Information</CardTitle>
              <CardDescription>
                Current server status and configuration details
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium">Server URL</p>
                    <p className="text-muted-foreground">
                      {localStorage.getItem('videoannotator_api_url') || 'http://localhost:8000'}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Token Status</p>
                    <Badge variant="outline">Configured</Badge>
                  </div>
                </div>
                
                <div className="pt-4 border-t">
                  <h4 className="font-medium mb-2">Server Debugging Tools</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Use these tools to test your server connection and debug API issues:
                  </p>
                  
                  <div className="space-y-2">
                    <div className="p-3 bg-muted rounded-md">
                      <p className="font-mono text-sm">python scripts/test_api_quick.py</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Comprehensive API endpoint testing
                      </p>
                    </div>
                    
                    <div className="p-3 bg-muted rounded-md">
                      <p className="font-mono text-sm">VideoAnnotatorDebug.runAllTests()</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Browser console debugging suite (paste in browser dev tools)
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="help" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Getting Help</CardTitle>
              <CardDescription>
                Resources and support for VideoAnnotator configuration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium">Common Issues</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground mt-2">
                    <li>Token authentication errors: Check your token is valid and not expired</li>
                    <li>Connection refused: Ensure VideoAnnotator server is running</li>
                    <li>404 errors: Server may not have all endpoints implemented yet</li>
                    <li>CORS issues: Check server allows requests from this domain</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-medium">Getting Your Token</h4>
                  <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground mt-2">
                    <li>Contact your VideoAnnotator server administrator</li>
                    <li>For development, use the default token: <code className="bg-muted px-1 rounded">dev-token</code></li>
                    <li>Check server documentation for token generation instructions</li>
                  </ol>
                </div>

                <div className="pt-4 border-t">
                  <Button variant="link" className="pl-0" asChild>
                    <a 
                      href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/CLIENT_TOKEN_SETUP_GUIDE.md"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      📖 View Complete Token Setup Guide
                    </a>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CreateSettings;