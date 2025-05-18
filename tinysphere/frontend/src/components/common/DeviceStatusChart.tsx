import { Box, Typography } from '@mui/material';
import React from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';

interface DeviceStatusChartProps {
  activeCount: number;
  inactiveCount: number;
}

const ACTIVE_COLOR = '#4CAF50';  // Green
const INACTIVE_COLOR = '#FFA500';  // Orange

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

  // Custom legend formatter for consistent styling
  const renderColorfulLegendText = (value: string) => {
    return <span style={{ color: '#111', fontSize: '0.875rem', paddingLeft: '4px' }}>{value}</span>;
  };

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
            fillOpacity={0.7}
          >
            <Cell fill={ACTIVE_COLOR} fillOpacity={0.7} />
            <Cell fill={INACTIVE_COLOR} fillOpacity={0.7} />
          </Pie>
          <Tooltip formatter={(value) => [`${value} devices`, 'Count']} />
          <Legend formatter={renderColorfulLegendText} />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default DeviceStatusChart;