// src/components/common/DriftEventsByDeviceChart.tsx
import React, { useState } from 'react';
import { Box, Button, Typography, CircularProgress } from '@mui/material';
import { 
  Area, 
  AreaChart, 
  CartesianGrid, 
  Legend, 
  ResponsiveContainer, 
  Tooltip, 
  XAxis, 
  YAxis 
} from 'recharts';
import ErrorDisplay from './ErrorDisplay';

// Define the data structure we expect
interface DriftTimelineData {
  date: string;
  [key: string]: any; // Device IDs as keys with count values
}

interface DriftEventsByDeviceChartProps {
  data: DriftTimelineData[];
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  currentTimeRange?: 'day' | 'week' | 'month';
  onTimeRangeChange?: (range: 'day' | 'week' | 'month') => void;
  height?: number | string;
}

// Array of colors to cycle through for device lines
const deviceColors = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', 
  '#82ca9d', '#ffc658', '#8dd1e1', '#a4de6c', '#d0ed57'
];

const DriftEventsByDeviceChart: React.FC<DriftEventsByDeviceChartProps> = ({
  data,
  isLoading = false,
  error = null,
  onRetry,
  currentTimeRange = 'week',
  onTimeRangeChange,
  height = 300
}) => {
  const [localTimeRange, setLocalTimeRange] = useState<'day' | 'week' | 'month'>(currentTimeRange);

  // Handler for time range changes
  const handleTimeRangeChange = (range: 'day' | 'week' | 'month') => {
    setLocalTimeRange(range);
    if (onTimeRangeChange) {
      onTimeRangeChange(range);
    }
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
      <Box sx={{ p: 4, textAlign: 'center', height }}>
        <Typography variant="body1" color="text.secondary">
          No drift events data available
        </Typography>
        {onRetry && (
          <Button onClick={onRetry} variant="outlined" sx={{ mt: 2 }}>
            Retry
          </Button>
        )}
      </Box>
    );
  }

  // Extract device IDs from the data (keys except 'date')
  const deviceIds = Object.keys(data[0]).filter(key => key !== 'date');

  return (
    <Box sx={{ height: typeof height === 'number' ? `${height}px` : height, p: 2 }}>
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
      
      <Box sx={{ height: `calc(${typeof height === 'number' ? height : '300'}px - 40px)` }}>
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
            
            {/* Create an Area for each device */}
            {deviceIds.map((deviceId, index) => (
              <Area
                key={deviceId}
                type="monotone"
                dataKey={deviceId}
                stackId="1"
                stroke={deviceColors[index % deviceColors.length]}
                fill={deviceColors[index % deviceColors.length]}
                name={deviceId}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default DriftEventsByDeviceChart;