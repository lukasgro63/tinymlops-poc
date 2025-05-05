// src/components/common/ErrorDisplay.tsx
import React from 'react';
import { Box, Button, CircularProgress, Typography } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

interface ErrorDisplayProps {
  error: string | null;
  loading?: boolean;
  onRetry?: () => void;
  height?: string | number;
}

/**
 * Standardized error display component
 */
const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ 
  error, 
  loading = false, 
  onRetry,
  height = '200px'
}) => {
  if (loading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: height 
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box 
        sx={{ 
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center', 
          alignItems: 'center', 
          height: height,
          textAlign: 'center',
          p: 2
        }}
      >
        <ErrorOutlineIcon 
          color="error" 
          sx={{ fontSize: 48, mb: 1 }} 
        />
        
        <Typography variant="h6" color="error" gutterBottom>
          Data Loading Error
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {error}
        </Typography>

        {onRetry && (
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={onRetry}
          >
            Retry
          </Button>
        )}
      </Box>
    );
  }
  
  return null;
};

export default ErrorDisplay;