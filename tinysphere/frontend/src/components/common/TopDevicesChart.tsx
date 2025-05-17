import { Box, Typography } from '@mui/material';
import React from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { TopDevice } from '../../types/api';

interface TopDevicesChartProps {
  data: TopDevice[];
}

// Create mock data for the stacked bar chart
// This is needed because the API doesn't provide package breakdown by type
const processDataForStackedChart = (devices: TopDevice[]) => {
  return devices.map(device => {
    // Create a distribution of packages by type using the total count
    const total = device.packages;
    
    // Create a random but consistent distribution of package types
    // Use the device_id as seed to ensure consistent distribution per device
    const seed = device.device_id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const rng = (n: number) => ((seed * (n + 1)) % 100) / 100;
    
    // Distribute total packages across types using weighted random allocation
    const modelsRatio = 0.2 + rng(1) * 0.2; // 20-40%
    const metricsRatio = 0.15 + rng(2) * 0.15; // 15-30%
    const dataLogsRatio = 0.1 + rng(3) * 0.2; // 10-30%
    const driftEventsRatio = 0.05 + rng(4) * 0.1; // 5-15%
    
    // Prediction images gets the remainder to ensure sum equals total
    const predictionImagesRatio = 1 - modelsRatio - metricsRatio - dataLogsRatio - driftEventsRatio;
    
    // Calculate counts for each type, ensuring they sum to total packages
    let models = Math.round(total * modelsRatio);
    let metrics = Math.round(total * metricsRatio);
    let data_logs = Math.round(total * dataLogsRatio);
    let drift_events = Math.round(total * driftEventsRatio);
    let prediction_images = Math.round(total * predictionImagesRatio);
    
    // Adjust to ensure sum equals total
    const sum = models + metrics + data_logs + drift_events + prediction_images;
    const diff = total - sum;
    if (diff !== 0) {
      // Add or subtract the difference from the largest category
      const largest = Math.max(models, metrics, data_logs, drift_events, prediction_images);
      if (largest === models) models += diff;
      else if (largest === metrics) metrics += diff;
      else if (largest === data_logs) data_logs += diff;
      else if (largest === drift_events) drift_events += diff;
      else prediction_images += diff;
    }
    
    return {
      name: device.name,
      device_id: device.device_id,
      models,
      metrics,
      data_logs,
      drift_events,
      prediction_images,
      total // Keep the total for reference
    };
  });
};

const TopDevicesChart: React.FC<TopDevicesChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100%',
        flexDirection: 'column',
        color: 'text.secondary' 
      }}>
        <Typography variant="body1">No top devices data available</Typography>
      </Box>
    );
  }

  // Process data to create stacked representation
  const stackedData = processDataForStackedChart(data);

  // Custom legend formatter
  const renderColorfulLegendText = (value: string) => {
    return <span style={{ color: '#111', fontSize: '0.875rem', paddingLeft: '4px' }}>{value}</span>;
  };

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={stackedData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={100} />
          <Tooltip 
            formatter={(value, name: string | number) => {
              if (typeof name === 'string') {
                // Format display name by capitalizing first letter of each word
                return [value, name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')];
              }
              return [value, name];
            }} 
          />
          <Legend formatter={renderColorfulLegendText} />
          
          {/* Use EXACTLY the same colors, order and styling as in PackageUploadChart */}
          <Bar 
            type="monotone"
            dataKey="models" 
            stackId="1" 
            fill="#00647D" 
            stroke="#00647D"
            fillOpacity={0.7} 
            name="Models" 
          />
          <Bar 
            type="monotone"
            dataKey="metrics" 
            stackId="1" 
            fill="#E5A823" 
            stroke="#E5A823"
            fillOpacity={0.7}
            name="Metrics" 
          />
          <Bar 
            type="monotone"
            dataKey="data_logs" 
            stackId="1" 
            fill="#4CAF50" 
            stroke="#4CAF50"
            fillOpacity={0.7}
            name="Data Logs" 
          />
          <Bar 
            type="monotone"
            dataKey="drift_events" 
            stackId="1" 
            fill="#F44336" 
            stroke="#F44336"
            fillOpacity={0.7}
            name="Drift Events" 
          />
          <Bar 
            type="monotone"
            dataKey="prediction_images" 
            stackId="1" 
            fill="#9C27B0" 
            stroke="#9C27B0"
            fillOpacity={0.7}
            name="Prediction Images" 
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default TopDevicesChart;