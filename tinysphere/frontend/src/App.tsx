// src/App.tsx
import { Box, Typography } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import MainLayout from './components/Layout/MainLayout';
import Dashboard from './pages/Dashboard';
import Devices from './pages/Devices';
import Models from './pages/Models';
import theme from './theme';

// Platzhalter-Komponenten für neue Seiten
const DevicesPage = () => (
  <Box sx={{ p: 0 }}>
    <Typography variant="h4" gutterBottom>Devices</Typography>
    <Typography variant="body1">
      Hier können Sie alle verbundenen TinyLCM-Geräte verwalten und überwachen.
    </Typography>
    <Box sx={{ mt: 4 }}>
      {/* Platzhalter für die tatsächliche Device-Liste */}
      <Typography variant="body2" color="text.secondary">
        Noch keine Geräte verbunden.
      </Typography>
    </Box>
  </Box>
);

const ModelsPage = () => (
  <Box sx={{ p: 0 }}>
    <Typography variant="h4" gutterBottom>Models</Typography>
    <Typography variant="body1">
      Hier können Sie alle Modelle verwalten, die auf Ihren TinyLCM-Geräten eingesetzt werden.
    </Typography>
    <Box sx={{ mt: 4 }}>
      {/* Platzhalter für die tatsächliche Modell-Liste */}
      <Typography variant="body2" color="text.secondary">
        Noch keine Modelle registriert.
      </Typography>
    </Box>
  </Box>
);

// External apps in iframes
const MLflowPage = () => (
  <iframe 
    src="http://localhost:5001" 
    title="MLflow" 
    style={{ width: '100%', height: 'calc(100vh - 16px)', border: 'none' }}
  />
);

const GrafanaPage = () => (
  <iframe 
    src="http://localhost:3000/grafana" 
    title="Grafana" 
    style={{ width: '100%', height: 'calc(100vh - 16px)', border: 'none' }}
  />
);

const MinioPage = () => (
  <iframe 
    src="http://localhost:9001" 
    title="MinIO" 
    style={{ width: '100%', height: 'calc(100vh - 16px)', border: 'none' }}
  />
);

const PostgresPage = () => (
  <iframe 
    src="http://localhost:5432" 
    title="PostgreSQL" 
    style={{ width: '100%', height: 'calc(100vh - 16px)', border: 'none' }}
  />
);

// Placeholder für Dokumentation
const DocumentationPage = () => (
  <div style={{ padding: 20 }}>
    <h1>Documentation</h1>
    <p>Diese Seite wird später mit ReadTheDocs oder einer ähnlichen Dokumentation eingebunden.</p>
  </div>
);

const ApiDocsPage = () => (
  <iframe 
    src="/docs" 
    title="API Documentation" 
    style={{ width: '100%', height: 'calc(100vh - 16px)', border: 'none' }}
  />
);

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          {/* Hauptnavigation */}
          <Route path="/" element={<MainLayout title="Dashboard"><Dashboard /></MainLayout>} />
          <Route path="/devices" element={<MainLayout title="Devices"><Devices /></MainLayout>} />
          <Route path="/models" element={<MainLayout title="Models"><Models /></MainLayout>} />
          
          {/* Anwendungen */}
          <Route path="/mlflow" element={
            <MainLayout title="MLflow" isEmbedded={true}>
              <MLflowPage />
            </MainLayout>
          } />
          <Route path="/grafana" element={
            <MainLayout title="Grafana" isEmbedded={true}>
              <GrafanaPage />
            </MainLayout>
          } />
          <Route path="/minio" element={
            <MainLayout title="MinIO" isEmbedded={true}>
              <MinioPage />
            </MainLayout>
          } />
          <Route path="/postgres" element={
            <MainLayout title="PostgreSQL" isEmbedded={true}>
              <PostgresPage />
            </MainLayout>
          } />
          
          {/* Dokumentation und Links */}
          <Route path="/documentation" element={<MainLayout title="Documentation"><DocumentationPage /></MainLayout>} />
          <Route path="/api-docs" element={
            <MainLayout title="API Documentation" isEmbedded={true}>
              <ApiDocsPage />
            </MainLayout>
          } />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;