// src/pages/About.tsx
import { Box, Paper, Typography } from '@mui/material';
import React from 'react';

const About: React.FC = () => {
  return (
    <Paper sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>
        About TinySphere
      </Typography>
      
      <Typography variant="body1" paragraph>
        TinySphere is a central MLOps platform for managing TinyLCM edge devices. It provides a unified infrastructure for model management, metrics tracking, and device monitoring.
      </Typography>
      
      <Typography variant="h5" gutterBottom sx={{ mt: 3 }}>
        Key Features
      </Typography>
      
      <Box component="ul" sx={{ pl: 2 }}>
        <Typography component="li" variant="body1">
          Centralized model management and versioning
        </Typography>
        <Typography component="li" variant="body1">
          Edge device monitoring and management
        </Typography>
        <Typography component="li" variant="body1">
          Metrics and logs collection from edge devices
        </Typography>
        <Typography component="li" variant="body1">
          MLOps workflow with MLflow integration
        </Typography>
        <Typography component="li" variant="body1">
          Data storage and management with MinIO
        </Typography>
      </Box>
      
      <Typography variant="h5" gutterBottom sx={{ mt: 3 }}>
        Architecture
      </Typography>
      
      <Typography variant="body1" paragraph>
        TinySphere consists of several integrated components:
      </Typography>
      
      <Box component="ul" sx={{ pl: 2 }}>
        <Typography component="li" variant="body1">
          <strong>TinySphere API</strong>: FastAPI-based backend for device and package management
        </Typography>
        <Typography component="li" variant="body1">
          <strong>MLflow</strong>: For experiment tracking and model registry
        </Typography>
        <Typography component="li" variant="body1">
          <strong>MinIO</strong>: S3-compatible storage for artifacts and data
        </Typography>
        <Typography component="li" variant="body1">
          <strong>PostgreSQL</strong>: Database for metadata and device information
        </Typography>
      </Box>
      
      <Box sx={{ mt: 4, bgcolor: '#f5f5f5', p: 2, borderRadius: 1 }}>
        <Typography variant="caption">
          Version 0.1.0 | &copy; 2025 | MIT License
        </Typography>
      </Box>
    </Paper>
  );
};

export default About;