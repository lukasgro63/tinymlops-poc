import { Box, Typography } from '@mui/material';
import React from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';

interface DeviceStatusChartProps {
  activeCount: number;
  inactiveCount: number;
}

const ACTIVE_COLOR = '#4CAF50';
const INACTIVE_COLOR = '#F44336';

const DeviceStatusChart: React.FC<DeviceStatusChartProps> = ({ activeCount, inactiveCount }) => {
  const data = [
    { name: 'Active', value: activeCount },
    { name: 'Inactive', value: inactiveCount }
  ];

  if (activeCount === 0 && inactiveCount === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100%',
        flexDirection: 'column',
        color: 'text.secondary' 
      }}>
        <Typography variant="body1">No device status data available</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            nameKey="name"
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            <Cell fill={ACTIVE_COLOR} />
            <Cell fill={INACTIVE_COLOR} />
          </Pie>
          <Tooltip formatter={(value) => [`${value} devices`, 'Count']} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default DeviceStatusChart;