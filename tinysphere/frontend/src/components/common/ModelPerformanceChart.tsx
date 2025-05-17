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

// Extended chart data interface
interface ChartDataItem {
  name: string;
  value: number | null;
  stage?: string;
  timestamp?: number;
  run_id?: string;
  isEmpty?: boolean; // Flag for empty data display
}

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
  
  // Filter visibility controlled from parent
  filtersVisible?: boolean;
  
  // Callback for when data is updated
  onLastUpdated?: (date: Date) => void;
}

const ModelPerformanceChart = React.forwardRef<{refresh: () => Promise<void>}, ModelPerformanceChartProps>(({
  models = [],
  performanceData = [],
  data = [],
  selectedMetric: externalSelectedMetric,
  onMetricChange,
  title,
  filtersVisible: externalFiltersVisible,
  onLastUpdated
}, ref) => {
  // Determine if we're in "models list" mode or "direct data" mode
  const isModelListMode = models.length > 0 && !data.length;

  // Internal state
  const [selectedModel, setSelectedModel] = useState<string>(models.length > 0 ? models[0] : '');
  const [selectedMetric, setSelectedMetric] = useState<string>(externalSelectedMetric || 'accuracy');
  const [chartData, setChartData] = useState<ChartDataItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Filter state
  const [internalFiltersVisible, setInternalFiltersVisible] = useState<boolean>(externalFiltersVisible !== undefined ? externalFiltersVisible : true);
  const [timeRange, setTimeRange] = useState<number>(7); // Default 7 days
  const [maxRuns, setMaxRuns] = useState<number>(10); // Default 10 runs
  
  // Use external filtersVisible if provided
  const filtersVisible = externalFiltersVisible !== undefined ? externalFiltersVisible : internalFiltersVisible;

  // Metric options - include all known operational metrics from MLflow
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([
    // ML metrics
    'accuracy', 'precision', 'recall', 'f1', 'loss',
    
    // Confidence metrics
    'confidence_mean', 'confidence_median', 'confidence_min', 'confidence_max', 
    'confidence_std', 'confidence_p5',
    
    // Latency metrics
    'latency_mean_ms', 'latency_median_ms', 'latency_min_ms', 'latency_max_ms',
    'latency_p95_ms', 'latency_p99_ms', 'latency_std_ms',
    'latency_mean', 'latency_median', 'latency_min', 'latency_max',
    'latency_p95', 'latency_p99', 'latency_std',
    
    // Operation metrics
    'total_inferences', 'uptime_seconds',
    
    // Distribution metrics (may contain NaN values)
    'prediction_distribution_negative', 'prediction_distribution_stone',
    'prediction_distribution_lego'
  ]);

  // Use data directly if provided
  useEffect(() => {
    if (data && data.length > 0) {
      // Pre-process data to handle NaN values
      const processedData = data.map(item => {
        const newItem = {...item};
        // Convert any NaN values to null so they're properly handled by the chart
        Object.keys(newItem).forEach(key => {
          const value = newItem[key];
          
          // Handle number NaN
          if (typeof value === 'number' && isNaN(value)) {
            newItem[key] = null;
          }
          
          // Handle string NaN values with explicit casting
          if (value !== null && typeof value === 'string') {
            const strValue = value as string;
            if (strValue.toLowerCase() === 'nan') {
              newItem[key] = null;
            }
          }
        });
        return newItem;
      });
      
      setChartData(processedData);
    }
  }, [data]);

  // Transform performance data for chart when model or metric changes in model list mode
  useEffect(() => {
    if (isModelListMode && selectedModel) {
      let filteredData: ModelPerformanceData[] = [];
      
      // Only filter if we have performance data
      if (performanceData.length > 0) {
        filteredData = performanceData.filter(item =>
          item.model_name === selectedModel &&
          item.metric_name === selectedMetric
        );
      }

      // Handle case where no data is available for this model/metric combination
      if (filteredData.length === 0) {
        console.warn(`No performance data found for model ${selectedModel} and metric ${selectedMetric}`);
        setChartData([
          { 
            name: 'No Data Available', 
            value: null,
            stage: '',
            isEmpty: true // Custom flag to identify empty data for rendering
          }
        ]);
        return;
      }

      // Transform for chart display
      const formattedData = filteredData.map(item => {
        // Per the interface, item.value should be number | null
        // But we'll handle any type safely
        let processedValue: number | null = null;
        
        if (item.value !== null) {
          if (typeof item.value === 'number') {
            // If it's a number but NaN, use null
            processedValue = isNaN(item.value) ? null : item.value;
          } else if (typeof item.value === 'string') {
            // If it's a string, check for NaN or try to parse it
            const str = item.value as string;
            if (str.toLowerCase() === 'nan') {
              processedValue = null;
            } else {
              const parsed = parseFloat(str);
              processedValue = isNaN(parsed) ? null : parsed;
            }
          }
          // All other types become null
        }
        
        return {
          name: `v${item.version}`,
          value: processedValue,
          stage: item.stage,
          timestamp: item.timestamp,
          run_id: item.run_id
        };
      });

      // Sort by version
      formattedData.sort((a, b) => {
        const versionA = parseInt(a.name.substring(1));
        const versionB = parseInt(b.name.substring(1));
        return versionA - versionB;
      });

      setChartData(formattedData);
    } else if (isModelListMode) {
      // No model selected
      setChartData([
        { 
          name: 'Select Model', 
          value: null,
          stage: '',
          isEmpty: true
        }
      ]);
    }
  }, [isModelListMode, selectedModel, selectedMetric, performanceData]);

  // Fetch new performance data when model, metric, or filters change
  useEffect(() => {
    if (isModelListMode) {
      fetchPerformanceData();
    }
  }, [isModelListMode, selectedModel, selectedMetric, timeRange, maxRuns]);

  // If external selectedMetric changes, update internal state
  useEffect(() => {
    if (externalSelectedMetric && externalSelectedMetric !== selectedMetric) {
      setSelectedMetric(externalSelectedMetric);
    }
  }, [externalSelectedMetric]);
  
  // Auto-select first available metric when availableMetrics changes and no metric is selected
  useEffect(() => {
    if (availableMetrics.length > 0 && (!selectedMetric || selectedMetric === 'accuracy')) {
      console.log(`Auto-selecting first available metric: ${availableMetrics[0]}`);
      setSelectedMetric(availableMetrics[0]);
      if (onMetricChange) {
        onMetricChange(availableMetrics[0]);
      }
    }
  }, [availableMetrics, selectedMetric, onMetricChange]);

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

      // Include operational metrics
      params.include_operational_metrics = true;

      // Fetch data with filters
      const data = await getModelsPerformance(params.metric, params.days, params.limit,
                                            params.model_name, undefined,
                                            params.include_operational_metrics);

      // Extract available metrics from the data
      const metricsSet = new Set<string>();
      
      // First try getting metrics from the performance data
      data.forEach(item => {
        if (item.metric_name) {
          metricsSet.add(item.metric_name);
        }
      });
      
      // If we found any metrics in the data, update the available metrics
      if (metricsSet.size > 0) {
        console.log(`Found ${metricsSet.size} metric types in performance data`);
        setAvailableMetrics(Array.from(metricsSet));
      } else {
        // Use default metrics as a fallback
        console.log("No metrics found in performance data, using defaults");
      }

      // Update last updated timestamp
      const updateDate = new Date();
      setLastUpdated(updateDate);
      
      // Notify parent of update if callback provided
      if (onLastUpdated) {
        onLastUpdated(updateDate);
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

  // Determine Y-axis domain based on metric
  const getYAxisDomain = (): [number, number | string] => {
    const metricToCheck = selectedMetric.toLowerCase();

    // Special handling for specific metrics
    if (metricToCheck.includes('latency') || metricToCheck.includes('time')) {
      return [0, 'auto'];  // For time-based metrics
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
  const formatTooltipValue = (value: any): string => {
    // Handle null, undefined or NaN values
    if (value === null || value === undefined || Number.isNaN(value)) {
      return 'N/A';
    }

    const metricToCheck = selectedMetric.toLowerCase();

    try {
      // Convert to number if string for consistent handling
      let numValue: number;
      
      if (typeof value === 'string') {
        // Explicitly cast to string to ensure TypeScript knows it's a string
        const strValue = value as string;
        
        // Check for string 'NaN' values
        if (strValue === 'NaN' || strValue === 'nan' || strValue.toLowerCase() === 'nan') {
          return 'N/A';
        }
        
        // Try to parse as number
        numValue = parseFloat(strValue);
        if (isNaN(numValue)) {
          return strValue; // Return original string if it can't be parsed
        }
      } else if (typeof value === 'number') {
        numValue = value as number;
      } else {
        return 'N/A'; // For any other type (like null, undefined, etc.)
      }

      // Special formatting based on metric type
      if (metricToCheck.includes('latency') || metricToCheck.includes('time')) {
        // Format as milliseconds with 2 decimal places
        return `${numValue.toFixed(2)} ms`;
      } else if (metricToCheck.includes('loss')) {
        // Format loss with 4 decimal places
        return numValue.toFixed(4);
      } else if (metricToCheck.includes('confidence')) {
        // Format confidence as percentage
        return `${(numValue * 100).toFixed(2)}%`;
      } else if (metricToCheck === 'total_inferences' || metricToCheck.includes('count')) {
        // Format counts as integers
        return Math.round(numValue).toString();
      } else if (metricToCheck.includes('distribution')) {
        // Format distribution values as percentages with 1 decimal place
        return `${(numValue * 100).toFixed(1)}%`;
      } else {
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

    // This is a fallback that should never be reached, but to satisfy TypeScript:
    return typeof value === 'string' ? value : String(value);
  };

  // Format metric name for display
  const formatMetricName = (metric: string): string => {
    // Replace underscores with spaces and capitalize each word
    return metric
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Expose refresh method via ref
  React.useImperativeHandle(ref, () => ({
    refresh: fetchPerformanceData
  }));

  // Filters section
  const renderFilters = () => {
    if (!filtersVisible) return null;

    return (
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 1 }}>
          {isModelListMode && (
            <Box sx={{ flex: '1 1 250px', minWidth: '200px' }}>
              <FormControl size="small" fullWidth>
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
            </Box>
          )}

          <Box sx={{ flex: isModelListMode ? '0 1 200px' : '1 1 250px', minWidth: isModelListMode ? '150px' : '200px' }}>
            <FormControl size="small" fullWidth>
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
          </Box>

          <Box sx={{ flex: '0 1 150px', minWidth: '120px' }}>
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

          <Box sx={{ flex: '0 1 150px', minWidth: '120px' }}>
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
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => {
              setTimeRange(7);
              setMaxRuns(10);
              
              // Only reset model if there are defaults available
              if (models.length > 0 && isModelListMode) {
                setSelectedModel(models[0]);
              }
              
              // Only reset metric if there are defaults available
              if (availableMetrics.length > 0) {
                setSelectedMetric(availableMetrics[0]);
                if (onMetricChange) {
                  onMetricChange(availableMetrics[0]);
                }
              }
            }}
            disabled={timeRange === 7 && maxRuns === 10}
          >
            Reset Filters
          </Button>
        </Box>
      </Box>
    );
  };

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      {/* Title only shown if within the component (not in SectionCard header) */}
      {title && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6">{title}</Typography>
        </Box>
      )}
      
      {/* Filters (toggleable) */}
      {(!externalSelectedMetric || isModelListMode) && renderFilters()}

      {/* Chart container */}
      <Box sx={{ height: filtersVisible ? 'calc(100% - 120px)' : 'calc(100% - 20px)' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>Loading model data...</Typography>
          </Box>
        ) : error ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : chartData.length === 0 || (chartData.length === 1 && chartData[0].isEmpty) ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>
              {chartData.length === 0 ? 
                'No performance data available for this model' : 
                chartData[0].name}
            </Typography>
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
                formatter={(value: any): [string, string] => [
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
                connectNulls={true}  // Skip null/undefined/NaN values in line connection
                isAnimationActive={false} // Disable animation to prevent issues with NaN values
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Box>
    </Box>
  );
});

export default ModelPerformanceChart;