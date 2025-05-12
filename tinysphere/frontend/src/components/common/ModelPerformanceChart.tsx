// src/components/common/ModelPerformanceChart.tsx
import {
  Box,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
  TextField,
  Typography,
  Tooltip as MuiTooltip,
  Button
} from '@mui/material';
import { FilterList, ExpandMore, ExpandLess, Refresh } from '@mui/icons-material';
import React, { useEffect, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getModelsPerformance } from '../../services/api';
import { ModelPerformanceData } from '../../types/api';

interface ModelPerformanceChartProps {
  // For Dashboard mode (fetches data internally)
  models?: string[];
  performanceData?: ModelPerformanceData[];

  // For Models page mode (data provided directly)
  data?: any[];

  // Control metric selection (optional - for external control)
  selectedMetric?: string;
  onMetricChange?: (metric: string) => void;

  // Optional title
  title?: string;
}

const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({
  models = [],
  performanceData = [],
  data = [],
  selectedMetric: externalSelectedMetric,
  onMetricChange,
  title
}) => {
  // Determine if we're in "models list" mode or "direct data" mode
  const isModelListMode = models.length > 0 && !data.length;

  // Internal state
  const [selectedModel, setSelectedModel] = useState<string>(models.length > 0 ? models[0] : '');
  const [selectedMetric, setSelectedMetric] = useState<string>(externalSelectedMetric || 'accuracy');
  const [chartData, setChartData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(false);
  const [timeRange, setTimeRange] = useState<number>(7); // Default 7 days
  const [maxRuns, setMaxRuns] = useState<number>(10); // Default 10 runs
  const [tags, setTags] = useState<string>(''); // Tags filter in key=value format

  // Metric options
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([
    'accuracy', 'precision', 'recall', 'f1', 'loss',
    'confidence_mean', 'latency_mean_ms', 'system_cpu_percent_avg', 'system_memory_percent_avg',
    'total_inferences'
  ]);

  // Use data directly if provided
  useEffect(() => {
    if (data && data.length > 0) {
      setChartData(data);
    }
  }, [data]);

  // Transform performance data for chart when model or metric changes in model list mode
  useEffect(() => {
    if (isModelListMode && selectedModel && performanceData.length > 0) {
      const filteredData = performanceData.filter(item =>
        item.model_name === selectedModel &&
        item.metric_name === selectedMetric
      );

      // Transform for chart display
      const formattedData = filteredData.map(item => ({
        name: `v${item.version}`,
        value: item.value,
        stage: item.stage,
        timestamp: item.timestamp,
        run_id: item.run_id
      }));

      // Sort by version
      formattedData.sort((a, b) => {
        const versionA = parseInt(a.name.substring(1));
        const versionB = parseInt(b.name.substring(1));
        return versionA - versionB;
      });

      setChartData(formattedData);
    }
  }, [isModelListMode, selectedModel, selectedMetric, performanceData]);

  // Fetch new performance data when model, metric, or filters change
  useEffect(() => {
    if (isModelListMode) {
      fetchPerformanceData();
    }
  }, [isModelListMode, selectedModel, selectedMetric, timeRange, maxRuns, tags]);

  // If external selectedMetric changes, update internal state
  useEffect(() => {
    if (externalSelectedMetric && externalSelectedMetric !== selectedMetric) {
      setSelectedMetric(externalSelectedMetric);
    }
  }, [externalSelectedMetric]);

  const fetchPerformanceData = async () => {
    if (!isModelListMode) return;

    setLoading(true);
    setError(null);

    try {
      // Build filter parameters
      const params: any = {
        metric: selectedMetric,
        days: timeRange,
        limit: maxRuns
      };

      // Add model filter if selected
      if (selectedModel) {
        params.model_name = selectedModel;
      }

      // Add tags filter if specified
      if (tags && tags.trim() !== '') {
        params.tags = tags;
      }

      // Include operational metrics
      params.include_operational_metrics = true;

      // Fetch data with filters
      const data = await getModelsPerformance(params.metric, params.days, params.limit,
                                            params.model_name, params.tags,
                                            params.include_operational_metrics);

      // Extract available metrics from the data
      const metricsSet = new Set<string>();
      data.forEach(item => {
        if (item.metric_name) {
          metricsSet.add(item.metric_name);
        }
      });

      // Update available metrics if we found any
      if (metricsSet.size > 0) {
        setAvailableMetrics(Array.from(metricsSet));
      }

      // Don't update performanceData here as we're using the prop
    } catch (error) {
      console.error('Error fetching model performance data:', error);
      setError('Failed to load model performance data');
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (event: SelectChangeEvent) => {
    setSelectedModel(event.target.value);
  };

  const handleMetricChange = (event: SelectChangeEvent) => {
    const metric = event.target.value;
    setSelectedMetric(metric);

    if (onMetricChange) {
      onMetricChange(metric);
    }
  };

  const handleTimeRangeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value);
    if (!isNaN(value) && value > 0) {
      setTimeRange(value);
    }
  };

  const handleMaxRunsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value);
    if (!isNaN(value) && value > 0) {
      setMaxRuns(value);
    }
  };

  const handleTagsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTags(event.target.value);
  };

  // Determine Y-axis domain based on metric
  const getYAxisDomain = (): [number, number | string] => {
    const metricToCheck = selectedMetric.toLowerCase();

    // Special handling for specific metrics
    if (metricToCheck.includes('latency') || metricToCheck.includes('time')) {
      return [0, 'auto'];  // For time-based metrics
    } else if (metricToCheck.includes('cpu') || metricToCheck.includes('memory')) {
      return [0, 100];     // For percentage-based metrics (0-100%)
    } else if (metricToCheck.includes('loss')) {
      return [0, 'auto'];  // For loss metrics
    } else if (metricToCheck.includes('confidence')) {
      return [0, 1];       // For confidence scores (0-1)
    } else if (metricToCheck === 'total_inferences' || metricToCheck.includes('count')) {
      return [0, 'auto'];  // For count metrics
    } else {
      // For other metrics like accuracy, precision, etc.
      return [0, 1];       // Typically between 0 and 1
    }
  };

  // Format the tooltip value
  const formatTooltipValue = (value: any) => {
    // Handle null, undefined or NaN values
    if (value === null || value === undefined || Number.isNaN(value)) {
      return 'N/A';
    }

    const metricToCheck = selectedMetric.toLowerCase();

    try {
      // Convert to number if string for consistent handling
      const numValue = typeof value === 'string' ? parseFloat(value) : value;

      // Special formatting based on metric type
      if (metricToCheck.includes('latency') || metricToCheck.includes('time')) {
        // Format as milliseconds with 2 decimal places
        return `${numValue.toFixed(2)} ms`;
      } else if (metricToCheck.includes('cpu') || metricToCheck.includes('memory')) {
        // Format as percentage
        return `${numValue.toFixed(1)}%`;
      } else if (metricToCheck.includes('loss')) {
        // Format loss with 4 decimal places
        return numValue.toFixed(4);
      } else if (metricToCheck.includes('confidence')) {
        // Format confidence as percentage
        return `${(numValue * 100).toFixed(2)}%`;
      } else if (metricToCheck === 'total_inferences' || metricToCheck.includes('count')) {
        // Format counts as integers
        return Math.round(numValue);
      } else if (typeof numValue === 'number') {
        // Standard metrics like accuracy are typically between 0-1
        if (numValue >= 0 && numValue <= 1) {
          return `${(numValue * 100).toFixed(2)}%`;
        } else {
          return numValue.toFixed(2);
        }
      }
    } catch (e) {
      console.warn(`Error formatting value "${value}":`, e);
      return 'Error';
    }

    return value;
  };

  // Format metric name for display
  const formatMetricName = (metric: string): string => {
    // Replace underscores with spaces and capitalize each word
    return metric
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Basic controls (model and metric)
  const renderBasicControls = () => {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {title && <Typography variant="h6">{title}</Typography>}
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          {isModelListMode && (
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                value={selectedModel}
                label="Model"
                onChange={handleModelChange}
              >
                {models.map((model) => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="metric-select-label">Metric</InputLabel>
            <Select
              labelId="metric-select-label"
              value={selectedMetric}
              label="Metric"
              onChange={handleMetricChange}
            >
              {availableMetrics.map((metric) => (
                <MenuItem key={metric} value={metric}>
                  {formatMetricName(metric)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <MuiTooltip title="Toggle Advanced Filters">
            <IconButton onClick={() => setShowAdvancedFilters(!showAdvancedFilters)} color="primary">
              {showAdvancedFilters ? <ExpandLess /> : <FilterList />}
            </IconButton>
          </MuiTooltip>

          <MuiTooltip title="Refresh Data">
            <IconButton onClick={fetchPerformanceData} color="primary">
              <Refresh />
            </IconButton>
          </MuiTooltip>
        </Box>
      </Box>
    );
  };

  // Advanced filters
  const renderAdvancedFilters = () => {
    if (!showAdvancedFilters) return null;

    return (
      <Box sx={{ mb: 3, p: 2, border: '1px solid #eee', borderRadius: 1 }}>
        <Typography variant="subtitle2" sx={{ mb: 2 }}>Advanced Filters</Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
            <TextField
              label="Time Range (days)"
              type="number"
              size="small"
              fullWidth
              value={timeRange}
              onChange={handleTimeRangeChange}
              InputProps={{ inputProps: { min: 1 } }}
            />
          </Box>

          <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
            <TextField
              label="Max Runs"
              type="number"
              size="small"
              fullWidth
              value={maxRuns}
              onChange={handleMaxRunsChange}
              InputProps={{ inputProps: { min: 1 } }}
            />
          </Box>

          <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
            <TextField
              label="Tags (key1=value1,key2=value2)"
              size="small"
              fullWidth
              value={tags}
              onChange={handleTagsChange}
              placeholder="e.g. device=pi01,type=test"
            />
          </Box>
        </Box>

        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => {
              setTimeRange(7);
              setMaxRuns(10);
              setTags('');
            }}
          >
            Reset Filters
          </Button>
        </Box>
      </Box>
    );
  };

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      {/* Basic controls (always shown) */}
      {(!externalSelectedMetric || isModelListMode) && renderBasicControls()}

      {/* Advanced filters (toggleable) */}
      {(!externalSelectedMetric || isModelListMode) && renderAdvancedFilters()}

      {/* Chart container */}
      <Box sx={{ height: showAdvancedFilters ? 'calc(100% - 170px)' : 'calc(100% - 60px)' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>Loading model data...</Typography>
          </Box>
        ) : error ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : chartData.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>No performance data available for this model</Typography>
          </Box>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={getYAxisDomain()} />
              <Tooltip
                formatter={(value) => [
                  formatTooltipValue(value),
                  formatMetricName(selectedMetric)
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey={data.length > 0 ? selectedMetric : "value"}
                stroke="#00647D"
                activeDot={{ r: 8 }}
                name={formatMetricName(selectedMetric)}
                connectNulls={true}  // Skip null/undefined values in line connection
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Box>
    </Box>
  );
};

export default ModelPerformanceChart;