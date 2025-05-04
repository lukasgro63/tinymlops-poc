// src/components/common/PackageUploadChart.tsx
import { Box, Button, Typography } from '@mui/material';
import React, { useState } from 'react';
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { PackageTimelineData } from '../../types/api';

interface PackageUploadChartProps {
  data: PackageTimelineData[];
  currentTimeRange?: 'day' | 'week' | 'month';
  onTimeRangeChange?: (range: 'day' | 'week' | 'month') => void;
}

const PackageUploadChart: React.FC<PackageUploadChartProps> = ({ 
  data, 
  currentTimeRange = 'week',
  onTimeRangeChange 
}) => {
  const [localTimeRange, setLocalTimeRange] = useState<'day' | 'week' | 'month'>(currentTimeRange);
  
  // Handler for time range changes
  const handleTimeRangeChange = (range: 'day' | 'week' | 'month') => {
    setLocalTimeRange(range);
    if (onTimeRangeChange) {
      onTimeRangeChange(range);
    }
  };
  
  if (!data || data.length === 0) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No package upload data available
        </Typography>
      </Box>
    );
  }
  
  return (
    <Box sx={{ height: '100%', p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Box sx={{ display: 'flex' }}>
          <Button 
            onClick={() => handleTimeRangeChange('day')}
            variant={localTimeRange === 'day' ? 'contained' : 'outlined'}
            color="primary"
            size="small"
            sx={{ borderRadius: '4px 0 0 4px' }}
          >
            Day
          </Button>
          <Button 
            onClick={() => handleTimeRangeChange('week')}
            variant={localTimeRange === 'week' ? 'contained' : 'outlined'}
            color="primary"
            size="small"
            sx={{ borderRadius: 0 }}
          >
            Week
          </Button>
          <Button 
            onClick={() => handleTimeRangeChange('month')}
            variant={localTimeRange === 'month' ? 'contained' : 'outlined'}
            color="primary"
            size="small"
            sx={{ borderRadius: '0 4px 4px 0' }}
          >
            Month
          </Button>
        </Box>
      </Box>
      
      <Box sx={{ height: 'calc(100% - 40px)' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="models" 
              stackId="1" 
              stroke="#00647D" 
              fill="#00647D" 
              name="Models"
            />
            <Area 
              type="monotone" 
              dataKey="metrics" 
              stackId="1" 
              stroke="#E5A823" 
              fill="#E5A823" 
              name="Metrics"
            />
            <Area 
              type="monotone" 
              dataKey="data_logs" 
              stackId="1" 
              stroke="#4CAF50" 
              fill="#4CAF50" 
              name="Data Logs"
            />
          </AreaChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default PackageUploadChart;