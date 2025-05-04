import { Box, Typography } from '@mui/material';
import React from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { DeviceTrend } from '../../types/api';

interface ConnectivityTrendChartProps {
  data: DeviceTrend[];
}

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
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="active" stroke="#4CAF50" name="Active Devices" />
          <Line type="monotone" dataKey="total" stroke="#0B2A5A" name="Total Devices" strokeDasharray="5 5" />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ConnectivityTrendChart;