// src/components/common/SectionCard.tsx
import { Box, Divider, Paper, Typography } from '@mui/material';
import React, { ReactNode } from 'react';

interface SectionCardProps {
  title: string;
  icon?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  height?: string | number;
  maxHeight?: string | number;
}

const SectionCard: React.FC<SectionCardProps> = ({
  title,
  icon,
  action,
  children,
  height,
  maxHeight
}) => {
  return (
    <Paper 
      elevation={1}
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: 2
      }}
    >
      <Box 
        sx={{ 
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          px: 3,
          py: 2
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {icon && <Box sx={{ mr: 1 }}>{icon}</Box>}
          <Typography variant="subtitle1" fontWeight="medium">{title}</Typography>
        </Box>
        {action}
      </Box>
      <Divider />
      <Box 
        sx={{ 
          flexGrow: 1, 
          height: height || 'auto',
          maxHeight: maxHeight || 'none',
          overflow: maxHeight ? 'auto' : 'visible',
          p: 0
        }}
      >
        {children}
      </Box>
    </Paper>
  );
};

export default SectionCard;