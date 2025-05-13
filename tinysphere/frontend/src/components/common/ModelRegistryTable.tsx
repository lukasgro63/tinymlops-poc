// src/components/common/ModelRegistryTable.tsx
import {
  Box,
  Chip,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Button,
  IconButton,
  Tooltip
} from '@mui/material';
import React, { useEffect, useState } from 'react';
import {
  CheckCircle as CheckCircleIcon,
  BugReport as BugReportIcon,
  Archive as ArchiveIcon,
  HourglassEmpty as HourglassEmptyIcon,
  OpenInNew as OpenInNewIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { ModelSummary } from '../../types/api';
import { getModelsSummary } from '../../services/api';

interface ModelRegistryTableProps {
  onModelSelect?: (modelName: string) => void;
  selectedModel?: string;
  refreshInterval?: number;
  initialData?: ModelSummary[];
}

const ModelRegistryTable: React.FC<ModelRegistryTableProps> = ({
  onModelSelect,
  selectedModel,
  refreshInterval = 0,
  initialData
}) => {
  const [loading, setLoading] = useState<boolean>(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [modelSummaries, setModelSummaries] = useState<ModelSummary[]>(initialData || []);

  // Function to fetch models data
  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getModelsSummary();
      setModelSummaries(data);
    } catch (err) {
      console.error('Error fetching model registry data:', err);
      setError('Failed to load model registry data');
    } finally {
      setLoading(false);
    }
  };

  // Initial data fetch if not provided
  useEffect(() => {
    if (!initialData) {
      fetchModels();
    }
  }, [initialData]);

  // Set up refresh interval if specified
  useEffect(() => {
    if (refreshInterval > 0) {
      const intervalId = setInterval(fetchModels, refreshInterval);
      return () => clearInterval(intervalId);
    }
  }, [refreshInterval]);

  // Handler for manual refresh
  const handleRefresh = () => {
    fetchModels();
  };

  // Handler for model selection
  const handleModelSelect = (modelName: string) => {
    if (onModelSelect) {
      onModelSelect(modelName);
    }
  };

  // Render stage chip
  const renderStageChip = (stage: string | undefined, version: number | undefined) => {
    if (!version) return '-';
    
    switch (stage) {
      case 'Production':
        return (
          <Chip 
            label={`v${version} - Production`} 
            color="success" 
            size="small" 
            icon={<CheckCircleIcon />} 
          />
        );
      case 'Staging':
        return (
          <Chip 
            label={`v${version} - Staging`} 
            color="warning" 
            size="small" 
            icon={<BugReportIcon />} 
          />
        );
      case 'Archived':
        return (
          <Chip 
            label={`v${version} - Archived`} 
            color="default" 
            size="small" 
            icon={<ArchiveIcon />} 
          />
        );
      default:
        return (
          <Chip 
            label={`v${version}`} 
            color="default" 
            size="small" 
            icon={<HourglassEmptyIcon />} 
          />
        );
    }
  };

  // Show loading state
  if (loading && !modelSummaries.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 3 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  // Show error state
  if (error && !modelSummaries.length) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="error" gutterBottom>
          {error}
        </Typography>
        <Button variant="outlined" size="small" onClick={handleRefresh}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" color="text.secondary">
          {modelSummaries.length} models in registry
        </Typography>
        
        <Tooltip title="Refresh model registry">
          <IconButton size="small" onClick={handleRefresh} disabled={loading}>
            {loading ? <CircularProgress size={18} /> : <RefreshIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
      </Box>
      
      <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>Model Name</TableCell>
              <TableCell align="center">Latest</TableCell>
              <TableCell align="center">Production</TableCell>
              <TableCell align="center">Staging</TableCell>
              <TableCell align="center">Total Versions</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {modelSummaries.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No models found in registry
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              modelSummaries.map((model) => (
                <TableRow 
                  key={model.name}
                  hover
                  onClick={() => handleModelSelect(model.name)}
                  selected={selectedModel === model.name}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell component="th" scope="row">
                    <Typography variant="body2" fontWeight={selectedModel === model.name ? 'bold' : 'normal'}>
                      {model.name}
                    </Typography>
                    {model.device_id && (
                      <Typography variant="caption" color="text.secondary" display="block">
                        Device: {model.device_id}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell align="center">
                    {renderStageChip(undefined, model.latest_version)}
                  </TableCell>
                  <TableCell align="center">
                    {model.has_production && model.production_version
                      ? renderStageChip('Production', model.production_version)
                      : '-'}
                  </TableCell>
                  <TableCell align="center">
                    {model.has_staging && model.staging_version
                      ? renderStageChip('Staging', model.staging_version)
                      : '-'}
                  </TableCell>
                  <TableCell align="center">
                    {model.total_versions}
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="View in MLflow">
                      <IconButton 
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          window.open(`/mlflow#/models/${model.name}`, '_blank');
                        }}
                      >
                        <OpenInNewIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ModelRegistryTable;