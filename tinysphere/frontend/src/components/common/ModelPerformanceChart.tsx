// src/components/common/ModelPerformanceChart.tsx
import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent, Typography } from '@mui/material';
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
}

const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({ 
  models = [], 
  performanceData = [],
  data = [],
  selectedMetric: externalSelectedMetric,
  onMetricChange
}) => {
  // Determine if we're in "models list" mode or "direct data" mode
  const isModelListMode = models.length > 0 && !data.length;
  
  // Internal state
  const [selectedModel, setSelectedModel] = useState<string>(models.length > 0 ? models[0] : '');
  const [selectedMetric, setSelectedMetric] = useState<string>(externalSelectedMetric || 'accuracy');
  const [chartData, setChartData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
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
        stage: item.stage
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
  
  // Fetch new performance data when model or metric changes
  useEffect(() => {
    if (isModelListMode) {
      fetchPerformanceData();
    }
  }, [isModelListMode, selectedModel, selectedMetric]);
  
  // If external selectedMetric changes, update internal state
  useEffect(() => {
    if (externalSelectedMetric && externalSelectedMetric !== selectedMetric) {
      setSelectedMetric(externalSelectedMetric);
    }
  }, [externalSelectedMetric]);
  
  const fetchPerformanceData = async () => {
    if (!isModelListMode || !selectedModel) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await getModelsPerformance(selectedMetric);
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
  
  // Determine Y-axis domain based on metric
  const getYAxisDomain = (): [number, number | string] => {
    const metricToCheck = selectedMetric.toLowerCase();
    if (metricToCheck.includes('loss')) {
      return [0, 'auto']; // For loss metrics, start at 0
    } else {
      // For other metrics like accuracy, precision, etc.
      return [0, 1];  // These are typically between 0 and 1
    }
  };
  
  // Format the tooltip value
  const formatTooltipValue = (value: any) => {
    if (value === null || value === undefined) return 'N/A';
    
    if (selectedMetric.toLowerCase().includes('loss')) {
      return value.toFixed(4);
    } else if (typeof value === 'number') {
      if (value >= 0 && value <= 1) {
        return `${(value * 100).toFixed(2)}%`;
      } else {
        return value.toFixed(4);
      }
    }
    return value;
  };
  
  const renderControls = () => {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mb: 2 }}>
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
            <MenuItem value="accuracy">Accuracy</MenuItem>
            <MenuItem value="precision">Precision</MenuItem>
            <MenuItem value="recall">Recall</MenuItem>
            <MenuItem value="f1">F1 Score</MenuItem>
            <MenuItem value="loss">Loss</MenuItem>
          </Select>
        </FormControl>
      </Box>
    );
  };
    
  return (
    <Box sx={{ height: '100%', p: 2 }}>
      {/* Only show controls if not externally controlled */}
      {(!externalSelectedMetric || isModelListMode) && renderControls()}
      
      <Box sx={{ height: 'calc(100% - 40px)' }}>
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
                  selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)
                ]}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey={data.length > 0 ? selectedMetric : "value"} 
                stroke="#00647D" 
                activeDot={{ r: 8 }} 
                name={selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)}
                connectNulls={true}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Box>
    </Box>
  );
};

export default ModelPerformanceChart;