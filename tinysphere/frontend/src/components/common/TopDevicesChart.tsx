import { Box, Typography } from '@mui/material';
import React from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { TopDevice } from '../../types/api';

interface TopDevicesChartProps {
  data: TopDevice[];
}

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

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={100} />
          <Tooltip />
          <Bar dataKey="packages" fill="#0B2A5A" name="Packages" />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default TopDevicesChart;