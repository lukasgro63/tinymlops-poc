import { Box, Typography } from '@mui/material';
import React from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { DeviceTrend } from '../../types/api';

// Erweiterte DeviceTrend Schnittstelle für zusätzliche Daten
interface ConnectivityTrendChartProps {
  data: (DeviceTrend & { inactive?: number })[];
}

// Wichtig: Dieselben Farben wie bei ActivityStatusChart verwenden
const ACTIVE_COLOR = '#4CAF50';  // Green
const INACTIVE_COLOR = '#FFA500';  // Orange - same as in device status chip
const TOTAL_COLOR = '#0B2A5A';  // Dark blue

const ConnectivityTrendChart: React.FC<ConnectivityTrendChartProps> = ({ data }) => {
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
        <Typography variant="body1">No connectivity trend data available</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip labelFormatter={(label) => `Date: ${label}`} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="active" 
            stroke={ACTIVE_COLOR} 
            name="Active Devices" 
            activeDot={{ r: 6 }}
          />
          <Line 
            type="monotone" 
            dataKey="inactive" 
            stroke={INACTIVE_COLOR} 
            name="Inactive Devices" 
          />
          <Line 
            type="monotone" 
            dataKey="total" 
            stroke={TOTAL_COLOR} 
            name="Total Devices" 
            strokeDasharray="5 5" 
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ConnectivityTrendChart;