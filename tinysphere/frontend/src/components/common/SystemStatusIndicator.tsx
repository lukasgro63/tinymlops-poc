// src/components/dashboard/SystemStatusIndicator.tsx
import { Box, Typography } from '@mui/material';
import React from 'react';

interface SystemStatusIndicatorProps {
  status: string;
}

const SystemStatusIndicator: React.FC<SystemStatusIndicatorProps> = ({ status }) => {
  // Status color based on value
  const getStatusColor = () => {
    switch (status.toLowerCase()) {
      case 'operational':
        return '#4CAF50'; // success
      case 'degraded':
        return '#FF9800'; // warning
      case 'error':
      case 'outage':
        return '#F44336'; // error
      default:
        return '#2196F3'; // info
    }
  };

  return (
    <Box 
      sx={{ 
        display: 'inline-flex', 
        alignItems: 'center', 
        px: 2, 
        py: 1, 
        mb: 3, 
        borderRadius: '20px',
        backgroundColor: `${getStatusColor()}15` 
      }}
    >
      <Box 
        sx={{ 
          width: 10, 
          height: 10, 
          borderRadius: '50%', 
          backgroundColor: getStatusColor(),
          mr: 1
        }} 
      />
      <Typography 
        variant="body2" 
        sx={{ fontWeight: 'medium', color: getStatusColor() }}
      >
        System Status: {status.toUpperCase()}
      </Typography>
    </Box>
  );
};

export default SystemStatusIndicator;