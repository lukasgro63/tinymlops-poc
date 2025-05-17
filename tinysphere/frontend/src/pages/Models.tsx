// src/pages/Models.tsx
import {
  Archive as ArchiveIcon,
  BugReport as BugReportIcon,
  CheckCircle as CheckCircleIcon,
  CompareArrows as CompareArrowsIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import StorageIcon from '@mui/icons-material/Storage';
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography
} from '@mui/material';
import React, { useEffect, useState } from 'react';
import ErrorDisplay from '../components/common/ErrorDisplay';
import ModelComparisonTable from '../components/common/ModelComparisonTable';
import ModelPerformanceChart from '../components/common/ModelPerformanceChart';
import ModelRegistryTable from '../components/common/ModelRegistryTable';
import SectionCard from '../components/common/SectionCard';
import {
  compareModelVersions,
  getModelMetrics,
  getModelVersions,
  getModels,
  getModelsSummary,
  getModelsPerformance
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
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  
  // Fetch initial data on load
  // Function to fetch all models data
  const fetchModelsData = async () => {
    try {
      setLoading(true);
      
      // Parallel requests for model list and summaries
      const [summaries, models] = await Promise.all([
        getModelsSummary(),
        getModels()
      ]);
      
      setModelSummaries(summaries);
      setModelList(models);
      
      // Select the first model if available and no model is currently selected
      if (models.length > 0 && !selectedModel) {
        setSelectedModel(models[0]);
      }
      
      // Update last updated timestamp
      setLastUpdated(new Date());
      
    } catch (err) {
      console.error('Error fetching model data:', err);
      setError('Failed to load model data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Fetch initial data on load
  useEffect(() => {
    fetchModelsData();
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
  
  // Fetch model metrics from MLflow using dashboard endpoint
  const fetchModelMetrics = async (modelName: string) => {
    try {
      setMetricsLoading(true);
      
      // First try using the dashboard performance endpoint which is more robust
      // This endpoint works better with experiment data
      console.log(`Fetching metrics for model ${modelName} using dashboard endpoint`);
      const performanceData = await getModelsPerformance(
        selectedMetric,  // Use current selected metric
        30,              // Get data for last 30 days
        100,             // Get up to 100 data points
        modelName,       // Filter to just this model
        undefined,       // No tag filtering
        true             // Include operational metrics
      );
      
      if (performanceData && performanceData.length > 0) {
        console.log(`Found ${performanceData.length} performance data points for model ${modelName}`);
        
        // Convert performance data format to version metrics format
        const versionMap = new Map();
        
        // Group metrics by version
        performanceData.forEach(item => {
          if (!versionMap.has(item.version)) {
            versionMap.set(item.version, {
              version: item.version,
              stage: item.stage,
              run_id: item.run_id || '',
              created_at: item.timestamp || Date.now(),
              metrics: {}
            });
          }
          
          // Add this metric to the version's metrics
          const versionData = versionMap.get(item.version);
          versionData.metrics[item.metric_name] = item.value;
        });
        
        // Convert map to array
        const processedMetrics = Array.from(versionMap.values());
        console.log(`Processed ${processedMetrics.length} versions with metrics`);
        
        setVersionMetrics(processedMetrics);
      } else {
        console.warn(`No performance data found, falling back to model metrics endpoint`);
        
        // Fallback to the original metrics endpoint
        const metrics = await getModelMetrics(modelName);
        
        if (metrics && metrics.length > 0) {
          console.log(`Found ${metrics.length} metrics from model endpoint`);
          setVersionMetrics(metrics);
        } else {
          console.warn(`No metrics found for model ${modelName} from either endpoint`);
          // Create empty metrics for existing versions
          if (modelVersions.length > 0) {
            const emptyMetrics = modelVersions.map(version => ({
              version: version.version,
              stage: version.stage,
              run_id: version.run_id || '',
              created_at: Date.now(),
              metrics: {}
            }));
            setVersionMetrics(emptyMetrics);
          } else {
            setVersionMetrics([]);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching model metrics:', err);
      setVersionMetrics([]);
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
  
  // Loading or error state
  if (loading || error) {
    return (
      <ErrorDisplay 
        error={error}
        loading={loading}
        onRetry={() => window.location.reload()}
        height="70vh"
      />
    );
  }
  
  return (
    <Box>
      {/* First row: Model Registry Table */}
      <Box sx={{ mb: 3 }}>
        <SectionCard 
          title="Model Registry" 
          icon={<StorageIcon style={{ fontSize: 20, color: '#00647D' }} />}
        >
          <ModelRegistryTable 
            initialData={modelSummaries}
            selectedModel={selectedModel}
            onModelSelect={(modelName) => setSelectedModel(modelName)}
          />
        </SectionCard>
      </Box>
      
      {/* Second row: Model Performance Trends */}
      <Box sx={{ mb: 3 }}>
        <SectionCard 
          title="Model Performance Trends"
          icon={<ShowChartIcon style={{ fontSize: 20, color: '#00647D' }} />}
          height={400}
        >
          {metricsLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <ModelPerformanceChart
              models={modelList}
              performanceData={versionMetrics.map(metric => {
                // Handle potential NaN values in metrics
                let metricValue: number | null = null;
                
                if (metric.metrics && metric.metrics[selectedMetric] !== undefined) {
                  const value = metric.metrics[selectedMetric];
                  
                  // Handle different value types
                  if (value === null) {
                    metricValue = null;
                  } else if (typeof value === 'number') {
                    // If it's a numeric NaN, set to null
                    metricValue = isNaN(value) ? null : value;
                  } else if (typeof value === 'string') {
                    // If it's a string 'NaN', set to null, otherwise try to parse it
                    const strValue = value as string;
                    if (strValue === 'NaN' || strValue === 'nan' || strValue.toLowerCase() === 'nan') {
                      metricValue = null;
                    } else {
                      // Try to parse as number
                      const parsed = parseFloat(strValue);
                      metricValue = isNaN(parsed) ? null : parsed;
                    }
                  } else {
                    // Any other type becomes null
                    metricValue = null;
                  }
                }
                
                return {
                  model_name: selectedModel,
                  version: metric.version,
                  stage: metric.stage,
                  metric_name: selectedMetric,
                  value: metricValue,
                  timestamp: metric.created_at,
                  run_id: metric.run_id
                };
              })}
              selectedMetric={selectedMetric}
              onMetricChange={(metric) => setSelectedMetric(metric)}
            />
          )}
        </SectionCard>
      </Box>
      
      {/* Third row: Model versions table */}
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
      
      {/* Fourth row: Model comparison (optional) */}
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