// src/components/common/DriftStatusByDeviceChart.tsx
import React, { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress, Button } from '@mui/material';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { DriveFileMove } from '@mui/icons-material';
import ErrorDisplay from './ErrorDisplay';
import { DriftStatus } from '../../types/api';

// Define the expected data structure
interface DeviceStatusData {
  device_id: string;
  name?: string; // Optional display name
  [key: string]: any; // For status counts: pending, validated, rejected, etc.
  total: number;
}

interface DriftStatusByDeviceChartProps {
  data?: DeviceStatusData[];
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  height?: number | string;
}

// Status color mapping - same as DriftStatisticsCard
const statusColors: Record<DriftStatus, string> = {
  pending: '#ff9800',
  validated: '#4caf50',
  rejected: '#f44336',
  resolved: '#2196f3',
  ignored: '#9e9e9e'
};

const DriftStatusByDeviceChart: React.FC<DriftStatusByDeviceChartProps> = ({
  data = [],
  isLoading = false,
  error = null,
  onRetry,
  height = 300
}) => {
  // Custom legend formatter
  const renderColorfulLegendText = (value: string) => {
    return <span style={{ color: '#111', fontSize: '0.875rem', paddingLeft: '4px' }}>{value}</span>;
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <ErrorDisplay
        error={error}
        loading={false}
        onRetry={onRetry}
        height={height}
      />
    );
  }

  if (!data || data.length === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height,
        flexDirection: 'column',
        color: 'text.secondary' 
      }}>
        <Typography variant="body1">No drift status data available</Typography>
        {onRetry && (
          <Button onClick={onRetry} variant="outlined" sx={{ mt: 2 }}>
            Retry
          </Button>
        )}
      </Box>
    );
  }

  // Sort data by total events descending
  const sortedData = [...data].sort((a, b) => b.total - a.total);

  return (
    <Box sx={{ height: height, p: 2 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={sortedData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="device_id" type="category" width={100} />
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
          
          {/* Status bars - use same colors as in DriftStatisticsCard */}
          <Bar 
            dataKey="pending" 
            stackId="status" 
            fill={statusColors.pending} 
            stroke={statusColors.pending}
            fillOpacity={0.7} 
            name="Pending" 
          />
          <Bar 
            dataKey="validated" 
            stackId="status" 
            fill={statusColors.validated} 
            stroke={statusColors.validated}
            fillOpacity={0.7}
            name="Validated" 
          />
          <Bar 
            dataKey="rejected" 
            stackId="status" 
            fill={statusColors.rejected} 
            stroke={statusColors.rejected}
            fillOpacity={0.7}
            name="Rejected" 
          />
          <Bar 
            dataKey="resolved" 
            stackId="status" 
            fill={statusColors.resolved} 
            stroke={statusColors.resolved}
            fillOpacity={0.7}
            name="Resolved" 
          />
          <Bar 
            dataKey="ignored" 
            stackId="status" 
            fill={statusColors.ignored} 
            stroke={statusColors.ignored}
            fillOpacity={0.7}
            name="Ignored" 
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default DriftStatusByDeviceChart;