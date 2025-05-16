// src/components/common/DevicePerformanceChart.tsx
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
import { getDevicePerformance, getDevicesSummary } from '../../services/api';
import { DevicePerformanceData, DeviceSummary } from '../../types/api';

interface DevicePerformanceChartProps {
  // Control device selection (optional - for external control)
  selectedDevice?: string;
  onDeviceChange?: (deviceId: string) => void;

  // Control metric selection (optional - for external control)
  selectedMetric?: string;
  onMetricChange?: (metric: string) => void;

  // Optional title
  title?: string;
}

const DevicePerformanceChart: React.FC<DevicePerformanceChartProps> = ({
  selectedDevice: externalSelectedDevice,
  onDeviceChange,
  selectedMetric: externalSelectedMetric,
  onMetricChange,
  title
}) => {
  // Internal state
  const [devices, setDevices] = useState<DeviceSummary[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>(externalSelectedDevice || '');
  const [selectedMetric, setSelectedMetric] = useState<string>(externalSelectedMetric || 'inference_time');
  const [performanceData, setPerformanceData] = useState<DevicePerformanceData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(false);
  const [timeRange, setTimeRange] = useState<number>(7); // Default 7 days
  const [maxDataPoints, setMaxDataPoints] = useState<number>(10); // Default 10 data points

  // Metric options - including all known operational metrics from MLflow
  const availableMetrics = [
    // Performance metrics
    'inference_time',
    'latency_mean_ms', 'latency_median_ms', 'latency_min_ms', 'latency_max_ms',
    'latency_p95_ms', 'latency_p99_ms', 'latency_std_ms',
    
    // System metrics
    'cpu_usage',
    'memory_usage',
    'system_cpu_percent_avg', 'system_cpu_percent_max', 'system_cpu_percent_current',
    'system_memory_percent_avg', 'system_memory_percent_max', 'system_memory_percent_current',
    
    // Operation metrics
    'total_inferences',
    'uptime_seconds'
  ];

  // Define state for available metrics based on device data
  const [deviceAvailableMetrics, setDeviceAvailableMetrics] = useState<string[]>([]);

  // Fetch device list on mount
  useEffect(() => {
    fetchDevicesList();
  }, []);

  // If external selectedDevice changes, update internal state
  useEffect(() => {
    if (externalSelectedDevice && externalSelectedDevice !== selectedDevice) {
      setSelectedDevice(externalSelectedDevice);
    }
  }, [externalSelectedDevice]);

  // If external selectedMetric changes, update internal state
  useEffect(() => {
    if (externalSelectedMetric && externalSelectedMetric !== selectedMetric) {
      setSelectedMetric(externalSelectedMetric);
    }
  }, [externalSelectedMetric]);

  // When device selection changes, update available metrics
  useEffect(() => {
    if (selectedDevice) {
      updateAvailableMetrics();
    }
  }, [selectedDevice]);

  // When device or metric selection changes, fetch new data
  useEffect(() => {
    if (selectedDevice && selectedMetric) {
      fetchDevicePerformance();
    }
  }, [selectedDevice, selectedMetric, timeRange, maxDataPoints]);

  // Fetch list of devices
  const fetchDevicesList = async () => {
    setLoading(true);
    setError(null);

    try {
      const deviceSummaries = await getDevicesSummary();
      setDevices(deviceSummaries);

      // Auto-select first device if none selected
      if (!selectedDevice && deviceSummaries.length > 0) {
        setSelectedDevice(deviceSummaries[0].device_id);
        if (onDeviceChange) {
          onDeviceChange(deviceSummaries[0].device_id);
        }
      }
    } catch (error) {
      console.error('Error fetching devices list:', error);
      setError('Failed to load devices list');
    } finally {
      setLoading(false);
    }
  };

  // Update available metrics for the selected device
  const updateAvailableMetrics = async () => {
    if (!selectedDevice) return;

    try {
      // Get device summary from API to check available metrics
      const deviceSummaries = await getDevicesSummary();
      const device = deviceSummaries.find(d => d.device_id === selectedDevice);
      
      if (!device || !device.mlflow_metrics) {
        // Default to showing all metrics if no MLflow metrics are available
        setDeviceAvailableMetrics([]);
        return;
      }
      
      // Find metrics that have valid data
      const validMetrics: string[] = [];
      
      // Check each metric in the available metrics list
      for (const metric of availableMetrics) {
        const metricData = device.mlflow_metrics[metric as keyof typeof device.mlflow_metrics];
        
        // Check if metric has valid data (not undefined, NaN, etc.)
        if (metricData && 
            typeof metricData === 'object' && 
            'avg' in metricData && 
            !isNaN(metricData.avg as number)) {
          validMetrics.push(metric);
        }
      }
      
      // Check if device's available_metrics field has anything
      if (device.mlflow_metrics.available_metrics && 
          Array.isArray(device.mlflow_metrics.available_metrics) && 
          device.mlflow_metrics.available_metrics.length > 0) {
        // Add metrics from available_metrics list that aren't already in our validMetrics
        device.mlflow_metrics.available_metrics.forEach(metric => {
          if (!validMetrics.includes(metric as string)) {
            validMetrics.push(metric as string);
          }
        });
      }
      
      // Update available metrics
      setDeviceAvailableMetrics(validMetrics);
      
      // If the currently selected metric is not in the valid list, select the first valid one
      if (validMetrics.length > 0 && !validMetrics.includes(selectedMetric)) {
        setSelectedMetric(validMetrics[0]);
        
        if (onMetricChange) {
          onMetricChange(validMetrics[0]);
        }
      }
    } catch (error) {
      console.error('Error updating available metrics:', error);
      // Default to showing all metrics if there's an error
      setDeviceAvailableMetrics([]);
    }
  };

  // Fetch device performance data
  const fetchDevicePerformance = async () => {
    if (!selectedDevice) return;

    setLoading(true);
    setError(null);

    try {
      // Get device performance data from API
      const data = await getDevicePerformance(
        selectedDevice,
        selectedMetric,
        timeRange,
        maxDataPoints
      );

      if (data.length === 0) {
        setError('No performance data available for this device and metric');
      } else {
        setPerformanceData(data);
      }
    } catch (error) {
      console.error('Error fetching device performance data:', error);
      setError('Failed to load device performance data');
    } finally {
      setLoading(false);
    }
  };

  const handleDeviceChange = (event: SelectChangeEvent) => {
    const deviceId = event.target.value;
    setSelectedDevice(deviceId);

    if (onDeviceChange) {
      onDeviceChange(deviceId);
    }
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

  const handleMaxDataPointsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value);
    if (!isNaN(value) && value > 0) {
      setMaxDataPoints(value);
    }
  };

  // Determine Y-axis domain based on metric
  const getYAxisDomain = (): [number, number | string] => {
    const metricToCheck = selectedMetric.toLowerCase();

    if (metricToCheck === 'inference_time' || metricToCheck.includes('latency')) {
      return [0, 'auto'];  // For time-based metrics (milliseconds)
    } else if (metricToCheck.includes('cpu') || metricToCheck.includes('memory')) {
      return [0, 100];     // For percentage-based metrics (0-100%)
    } else if (metricToCheck.includes('total_inferences')) {
      return [0, 'auto'];  // For count metrics
    }

    return [0, 'auto']; // Default case
  };

  // Format the tooltip value
  const formatTooltipValue = (value: any): string => {
    // Handle null, undefined or NaN values
    if (value === null || value === undefined || Number.isNaN(value)) {
      return 'N/A';
    }

    const metricToCheck = selectedMetric.toLowerCase();
    const numValue = Number(value);

    if (metricToCheck === 'inference_time' || metricToCheck.includes('latency')) {
      return `${numValue.toFixed(2)} ms`;
    } else if (metricToCheck.includes('cpu') || metricToCheck.includes('memory')) {
      return `${numValue.toFixed(1)}%`;
    } else if (metricToCheck === 'total_inferences') {
      return Math.round(numValue).toString();
    } else if (metricToCheck === 'uptime_seconds') {
      // Format seconds into readable time
      const hours = Math.floor(numValue / 3600);
      const minutes = Math.floor((numValue % 3600) / 60);
      const seconds = Math.floor(numValue % 60);
      
      if (hours > 0) {
        return `${hours}h ${minutes}m ${seconds}s`;
      } else if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
      } else {
        return `${seconds}s`;
      }
    }

    return numValue.toFixed(2);
  };

  // Format metric name for display
  const formatMetricName = (metric: string): string => {
    return metric
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Format timestamp for X-axis
  const formatXAxis = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleDateString();
  };

  // Get device name from ID
  const getDeviceDisplayName = (deviceId: string): string => {
    const device = devices.find(d => d.device_id === deviceId);
    return device?.hostname || deviceId;
  };

  // Basic controls (device and metric selectors)
  const renderBasicControls = () => {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {title && <Typography variant="h6">{title}</Typography>}
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="device-select-label">Device</InputLabel>
            <Select
              labelId="device-select-label"
              value={selectedDevice}
              label="Device"
              onChange={handleDeviceChange}
            >
              {devices.map((device) => (
                <MenuItem key={device.device_id} value={device.device_id}>
                  {device.hostname || device.device_id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="metric-select-label">Metric</InputLabel>
            <Select
              labelId="metric-select-label"
              value={selectedMetric}
              label="Metric"
              onChange={handleMetricChange}
            >
              {/* Only show metrics that are available for this device */}
              {deviceAvailableMetrics.length > 0 
                ? deviceAvailableMetrics.map((metric) => (
                    <MenuItem key={metric} value={metric}>
                      {formatMetricName(metric)}
                    </MenuItem>
                  ))
                : availableMetrics.map((metric) => (
                    <MenuItem key={metric} value={metric}>
                      {formatMetricName(metric)}
                    </MenuItem>
                  ))
              }
            </Select>
          </FormControl>

          <MuiTooltip title="Toggle Advanced Filters">
            <IconButton onClick={() => setShowAdvancedFilters(!showAdvancedFilters)} color="primary">
              {showAdvancedFilters ? <ExpandLess /> : <FilterList />}
            </IconButton>
          </MuiTooltip>

          <MuiTooltip title="Refresh Data">
            <IconButton onClick={fetchDevicePerformance} color="primary">
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
              label="Max Data Points"
              type="number"
              size="small"
              fullWidth
              value={maxDataPoints}
              onChange={handleMaxDataPointsChange}
              InputProps={{ inputProps: { min: 1 } }}
            />
          </Box>
        </Box>

        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => {
              setTimeRange(7);
              setMaxDataPoints(10);
            }}
          >
            Reset Filters
          </Button>
        </Box>
      </Box>
    );
  };

  // Get chart color based on metric
  const getChartColor = (): string => {
    const metricToCheck = selectedMetric.toLowerCase();
    
    if (metricToCheck === 'inference_time' || metricToCheck.includes('latency')) {
      return '#00647D'; // Blue
    } else if (metricToCheck.includes('cpu')) {
      return '#E5A823'; // Yellow
    } else if (metricToCheck.includes('memory')) {
      return '#4CAF50'; // Green
    } else {
      return '#00647D'; // Default blue
    }
  };

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      {/* Basic controls (always shown) */}
      {renderBasicControls()}

      {/* Advanced filters (toggleable) */}
      {renderAdvancedFilters()}

      {/* Chart container */}
      <Box sx={{ height: showAdvancedFilters ? 'calc(100% - 170px)' : 'calc(100% - 60px)' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>Loading device performance data...</Typography>
          </Box>
        ) : error ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : performanceData.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>
              No performance data available for this device
            </Typography>
          </Box>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={performanceData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={formatXAxis}
                label={{ value: 'Date', position: 'insideBottomRight', offset: -10 }}
              />
              <YAxis domain={getYAxisDomain()} />
              <Tooltip
                labelFormatter={(timestamp) => `Date: ${new Date(timestamp).toLocaleDateString()}`}
                formatter={(value: any): [string, string] => [
                  formatTooltipValue(value),
                  formatMetricName(selectedMetric)
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke={getChartColor()}
                activeDot={{ r: 8 }}
                name={formatMetricName(selectedMetric)}
                connectNulls={true}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Box>
    </Box>
  );
};

export default DevicePerformanceChart;