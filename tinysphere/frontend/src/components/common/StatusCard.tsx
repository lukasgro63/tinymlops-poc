// src/components/common/StatusCard.tsx
import { Box, Paper, Typography, Tooltip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import React from 'react';

interface StatusCardProps {
  title: string;
  value: string | number;
  secondaryValue?: string;
  secondaryLabel?: string;
  icon: React.ReactNode;
  color?: string;
  onClick?: () => void;
  tooltip?: string;
}

const StatusCard: React.FC<StatusCardProps> = ({
  title,
  value,
  secondaryValue,
  secondaryLabel,
  icon,
  color = '#00647D',
  onClick,
  tooltip
}) => {
  return (
    <Paper
      elevation={1}
      sx={{
        p: 3,
        height: '100%',
        borderRadius: 2,
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s',
        '&:hover': onClick ? {
          transform: 'translateY(-4px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
        } : {}
      }}
      onClick={onClick}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            {tooltip && (
              <Tooltip title={tooltip}>
                <InfoIcon fontSize="small" sx={{ ml: 0.5, fontSize: 16, color: 'text.secondary' }} />
              </Tooltip>
            )}
          </Box>
          <Typography variant="h4" sx={{ mt: 1, fontWeight: 'medium' }}>
            {value}
          </Typography>

          {secondaryValue && (
            <Typography variant="caption" color="text.secondary">
              {secondaryLabel || 'Total'}: {secondaryValue}
            </Typography>
          )}
        </Box>
        
        <Box 
          sx={{ 
            backgroundColor: `${color}15`,
            p: 1.5,
            borderRadius: '50%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          {icon}
        </Box>
      </Box>
    </Paper>
  );
};

export default StatusCard;