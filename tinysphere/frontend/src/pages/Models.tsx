// src/pages/Models.tsx
import {
  Archive as ArchiveIcon,
  BugReport as BugReportIcon,
  CheckCircle as CheckCircleIcon,
  CompareArrows as CompareArrowsIcon,
  HourglassEmpty as HourglassEmptyIcon
} from '@mui/icons-material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';
import React, { useEffect, useState } from 'react';
import ModelComparisonTable from '../components/common/ModelComparisonTable';
import ModelPerformanceChart from '../components/common/ModelPerformanceChart';
import SectionCard from '../components/common/SectionCard';
import {
  compareModelVersions,
  getModelMetrics,
  getModelVersions,
  getModels,
  getModelsSummary
} from '../services/api';
import {
  MetricRow,
  Model,
  ModelSummary,
  ModelVersionMetrics
} from '../types/api';

const Models: React.FC = () => {
  // State for model data
  const [modelSummaries, setModelSummaries] = useState<ModelSummary[]>([]);
  const [modelList, setModelList] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [modelVersions, setModelVersions] = useState<Model[]>([]);
  const [selectedVersions, setSelectedVersions] = useState<number[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('accuracy');
  
  // Performance and comparison data
  const [versionMetrics, setVersionMetrics] = useState<ModelVersionMetrics[]>([]);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [comparisonData, setComparisonData] = useState<MetricRow[]>([]);
  
  // Loading and error states
  const [loading, setLoading] = useState<boolean>(true);
  const [metricsLoading, setMetricsLoading] = useState<boolean>(false);
  const [comparisonLoading, setComparisonLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Fetch initial data on load
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Parallel requests for model list and summaries
        const [summaries, models] = await Promise.all([
          getModelsSummary(),
          getModels()
        ]);
        
        setModelSummaries(summaries);
        setModelList(models);
        
        // Select the first model if available
        if (models.length > 0) {
          setSelectedModel(models[0]);
        }
        
      } catch (err) {
        console.error('Error fetching model data:', err);
        setError('Failed to load model data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);
  
  // Fetch versions when selected model changes
  useEffect(() => {
    if (selectedModel) {
      fetchModelVersions(selectedModel);
    }
  }, [selectedModel]);
  
  // Fetch metrics when selected model or metric changes
  useEffect(() => {
    if (selectedModel) {
      fetchModelMetrics(selectedModel);
    }
  }, [selectedModel, selectedMetric]);
  
  // Fetch comparison data when selected versions change
  useEffect(() => {
    if (selectedModel && selectedVersions.length > 0) {
      fetchComparisonData(selectedModel, selectedVersions);
    } else {
      setComparisonData([]);
    }
  }, [selectedModel, selectedVersions]);
  
  // Transform metrics data for the chart
  useEffect(() => {
    if (versionMetrics.length > 0) {
      const chartData = versionMetrics.map(version => {
        // Find the requested metric, default to the first available if not found
        let metricValue = null;
        if (version.metrics && Object.keys(version.metrics).length > 0) {
          metricValue = version.metrics[selectedMetric] !== undefined 
            ? version.metrics[selectedMetric] 
            : Object.values(version.metrics)[0];
        }
        
        return {
          name: `v${version.version}`,
          [selectedMetric]: metricValue,
          stage: version.stage
        };
      });
      
      // Sort by version
      chartData.sort((a, b) => {
        const versionA = parseInt(a.name.substring(1));
        const versionB = parseInt(b.name.substring(1));
        return versionA - versionB;
      });
      
      setPerformanceData(chartData);
    } else {
      setPerformanceData([]);
    }
  }, [versionMetrics, selectedMetric]);
  
  // Fetch model versions
  const fetchModelVersions = async (modelName: string) => {
    try {
      const versions = await getModelVersions(modelName);
      setModelVersions(versions);
      
      // Select the latest two versions for comparison
      if (versions.length > 0) {
        // Sort by version descending
        const sortedVersions = [...versions].sort((a, b) => b.version - a.version);
        setSelectedVersions([
          sortedVersions[0].version,
          sortedVersions.length > 1 ? sortedVersions[1].version : sortedVersions[0].version
        ]);
      }
    } catch (err) {
      console.error(`Error fetching versions for model ${modelName}:`, err);
      setError(`Failed to load versions for model ${modelName}.`);
    }
  };
  
  // Fetch model metrics from MLflow
  const fetchModelMetrics = async (modelName: string) => {
    try {
      setMetricsLoading(true);
      const metrics = await getModelMetrics(modelName);
      setVersionMetrics(metrics);
    } catch (err) {
      console.error('Error fetching model metrics:', err);
      // Don't set global error for just the metrics
    } finally {
      setMetricsLoading(false);
    }
  };
  
  // Fetch comparison data for selected versions
  const fetchComparisonData = async (modelName: string, versions: number[]) => {
    if (versions.length === 0) return;
    
    try {
      setComparisonLoading(true);
      const data = await compareModelVersions(modelName, versions);
      setComparisonData(data);
    } catch (err) {
      console.error('Error fetching comparison data:', err);
      // Don't set global error for just the comparison
    } finally {
      setComparisonLoading(false);
    }
  };
  
  // Render stage chip for model versions
  const renderStageChip = (stage: string) => {
    switch (stage) {
      case 'Production':
        return <Chip label="Production" color="success" size="small" icon={<CheckCircleIcon />} />;
      case 'Staging':
        return <Chip label="Staging" color="warning" size="small" icon={<BugReportIcon />} />;
      case 'Archived':
        return <Chip label="Archived" color="default" size="small" icon={<ArchiveIcon />} />;
      default:
        return <Chip label="None" color="default" size="small" icon={<HourglassEmptyIcon />} />;
    }
  };
  
  // Format creation timestamp
  const formatTimestamp = (timestamp: number) => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  // Loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // Error state
  if (error) {
    return (
      <Box>
        <Typography color="error" variant="h6" gutterBottom>
          Error
        </Typography>
        <Typography>{error}</Typography>
        <Button 
          variant="contained" 
          sx={{ mt: 2 }} 
          onClick={() => window.location.reload()}
        >
          Reload
        </Button>
      </Box>
    );
  }
  
  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Machine Learning Models</Typography>
        <Button 
          variant="contained" 
          color="primary"
          onClick={() => window.open('/mlflow#/models/register', '_blank')}
        >
          Register New Model
        </Button>
      </Box>
      
      {/* First row: Performance chart and Registry status */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
        {/* Model performance chart */}
        <Box sx={{ flex: '1 1 600px', minWidth: '300px' }}>
          <SectionCard 
            title="Model Performance Trends"
            icon={<ShowChartIcon style={{ fontSize: 20, color: '#00647D' }} />}
          >
            <Box sx={{ height: 300 }}>
              {metricsLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress size={24} />
                </Box>
              ) : (
                <ModelPerformanceChart 
                  models={modelList}
                  selectedMetric={selectedMetric}
                  onMetricChange={(metric) => setSelectedMetric(metric)}
                  performanceData={versionMetrics.map(metric => ({
                    model_name: selectedModel,
                    version: metric.version,
                    stage: metric.stage,
                    metric_name: selectedMetric,
                    value: metric.metrics[selectedMetric] || 0,
                    timestamp: metric.created_at
                  }))}
                />
              )}
            </Box>
          </SectionCard>
        </Box>
        
        {/* Registry status */}
        <Box sx={{ flex: '1 1 400px', minWidth: '300px' }}>
          <SectionCard title="Model Registry Status">
            <TableContainer component={Paper} sx={{ maxHeight: 350, overflow: 'auto' }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Model Name</TableCell>
                    <TableCell align="center">Latest</TableCell>
                    <TableCell align="center">Production</TableCell>
                    <TableCell align="center">Staging</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {modelSummaries.map((model) => (
                    <TableRow 
                      key={model.name}
                      hover
                      onClick={() => setSelectedModel(model.name)}
                      selected={selectedModel === model.name}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" fontWeight={selectedModel === model.name ? 'bold' : 'normal'}>
                          {model.name}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">{model.latest_version}</TableCell>
                      <TableCell align="center">
                        {model.production_version || '-'}
                      </TableCell>
                      <TableCell align="center">
                        {model.staging_version || '-'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </SectionCard>
        </Box>
      </Box>
      
      {/* Second row: Model versions table */}
      <Box sx={{ mb: 3 }}>
        <SectionCard title={`${selectedModel} Versions`}>
          <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Version</TableCell>
                  <TableCell>Stage</TableCell>
                  <TableCell>Source</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelVersions.map((version) => {
                  // Find creation timestamp from metrics data if available
                  const versionMetric = versionMetrics.find(m => m.version === version.version);
                  const creationTime = versionMetric?.created_at;
                  
                  return (
                    <TableRow 
                      key={`${version.name}-${version.version}`}
                      hover
                      selected={selectedVersions.includes(version.version)}
                    >
                      <TableCell>{version.version}</TableCell>
                      <TableCell>{renderStageChip(version.stage)}</TableCell>
                      <TableCell sx={{ maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {version.source || '-'}
                      </TableCell>
                      <TableCell>
                        {creationTime ? formatTimestamp(creationTime) : 'Unknown'}
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                          <Button 
                            size="small" 
                            variant="outlined"
                            onClick={() => {
                              if (selectedVersions.includes(version.version)) {
                                setSelectedVersions(selectedVersions.filter(v => v !== version.version));
                              } else if (selectedVersions.length < 2) {
                                setSelectedVersions([...selectedVersions, version.version]);
                              } else {
                                setSelectedVersions([selectedVersions[1], version.version]);
                              }
                            }}
                          >
                            {selectedVersions.includes(version.version) ? 'Deselect' : 'Compare'}
                          </Button>
                          <Button 
                            size="small" 
                            variant="contained"
                            onClick={() => window.open(`/mlflow#/models/${version.name}/versions/${version.version}`, '_blank')}
                          >
                            Details
                          </Button>
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </SectionCard>
      </Box>
      
      {/* Third row: Model comparison (optional) */}
      {selectedVersions.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <SectionCard 
            title={`Model Comparison`}
            icon={<CompareArrowsIcon sx={{ fontSize: 20, color: '#00647D' }} />}
          >
            {comparisonLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
                <CircularProgress size={24} />
              </Box>
            ) : (
              <ModelComparisonTable 
                modelName={selectedModel}
                versions={selectedVersions}
                comparisonData={comparisonData}
              />
            )}
          </SectionCard>
        </Box>
      )}
    </Box>
  );
};

export default Models;