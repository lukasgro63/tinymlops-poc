import { Box, Typography } from '@mui/material';
import React from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { PlatformDistribution } from '../../types/api';

interface PlatformDistributionChartProps {
  data: PlatformDistribution[];
}

const COLORS = ['#0B2A5A', '#4CAF50', '#F44336', '#FF9800', '#E5A823', '#2196F3', '#9C27B0'];

const PlatformDistributionChart: React.FC<PlatformDistributionChartProps> = ({ data }) => {
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
        <Typography variant="body1">No platform data available</Typography>
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
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} fillOpacity={0.7} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => [`${value} devices`, 'Count']} />
          <Legend formatter={renderColorfulLegendText} />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PlatformDistributionChart;