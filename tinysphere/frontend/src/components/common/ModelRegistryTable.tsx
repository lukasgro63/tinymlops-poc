// src/components/common/ModelRegistryTable.tsx
import {
  Box,
  Chip,
  CircularProgress,
  FormControl,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  SelectChangeEvent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TableSortLabel,
  TextField,
  Tooltip,
  Typography,
  Button
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  BugReport as BugReportIcon,
  Archive as ArchiveIcon,
  HourglassEmpty as HourglassEmptyIcon,
  OpenInNew as OpenInNewIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  FilterList as FilterListIcon
} from '@mui/icons-material';
import React, { useEffect, useState, useMemo } from 'react';
import { ModelSummary } from '../../types/api';
import { getModelsSummary } from '../../services/api';

// Type for sorting
type Order = 'asc' | 'desc';
type SortableKeys = 'name' | 'latest_version' | 'total_versions';

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
  // Data state
  const [loading, setLoading] = useState<boolean>(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [modelSummaries, setModelSummaries] = useState<ModelSummary[]>(initialData || []);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  
  // Filter state
  const [filtersVisible, setFiltersVisible] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [stageFilter, setStageFilter] = useState<string>('all');
  const [deviceFilter, setDeviceFilter] = useState<string>('all');
  
  // Sorting and pagination
  const [order, setOrder] = useState<Order>('desc');
  const [orderBy, setOrderBy] = useState<SortableKeys>('latest_version');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Function to fetch models data
  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getModelsSummary();
      setModelSummaries(data);
      setLastUpdated(new Date());
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
    } else {
      setLastUpdated(new Date());
    }
  }, [initialData]);

  // Set up refresh interval if specified
  useEffect(() => {
    if (refreshInterval > 0) {
      const intervalId = setInterval(fetchModels, refreshInterval);
      return () => clearInterval(intervalId);
    }
  }, [refreshInterval]);

  // Filter and sort models
  const filteredModels = useMemo(() => {
    if (!modelSummaries.length) return [];
    
    let filtered = [...modelSummaries];
    
    // Apply search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter(model => 
        model.name.toLowerCase().includes(searchLower) ||
        (model.device_id && model.device_id.toLowerCase().includes(searchLower))
      );
    }
    
    // Apply stage filter
    if (stageFilter !== 'all') {
      if (stageFilter === 'production') {
        filtered = filtered.filter(model => model.has_production);
      } else if (stageFilter === 'staging') {
        filtered = filtered.filter(model => model.has_staging);
      } else if (stageFilter === 'none') {
        filtered = filtered.filter(model => !model.has_production && !model.has_staging);
      }
    }
    
    // Apply device filter if selected
    if (deviceFilter !== 'all') {
      filtered = filtered.filter(model => model.device_id === deviceFilter);
    }
    
    // Apply sorting
    if (orderBy) {
      filtered.sort((a, b) => {
        let aValue: any = a[orderBy];
        let bValue: any = b[orderBy];
        
        // Special handling for string values
        if (typeof aValue === 'string' && typeof bValue === 'string') {
          aValue = aValue.toLowerCase();
          bValue = bValue.toLowerCase();
        }
        
        if (aValue === bValue) return 0;
        
        const comparison = aValue < bValue ? -1 : 1;
        return order === 'asc' ? comparison : -comparison;
      });
    }
    
    return filtered;
  }, [modelSummaries, searchTerm, stageFilter, deviceFilter, orderBy, order]);
  
  // Get unique device IDs for filter
  const uniqueDevices = useMemo(() => {
    const devices = new Set<string>();
    modelSummaries.forEach(model => {
      if (model.device_id) {
        devices.add(model.device_id);
      }
    });
    return Array.from(devices);
  }, [modelSummaries]);

  // Handler for sort request
  const handleRequestSort = (property: SortableKeys) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  // Pagination handlers
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

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
    <Box sx={{ p: 2 }}>
      {/* Filter controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography variant="subtitle1">Models</Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Last updated">
            <Typography variant="caption" sx={{ alignSelf: 'center', mr: 1, color: 'text.secondary' }}>
              {lastUpdated ? `Updated: ${lastUpdated.toLocaleTimeString()}` : ''}
            </Typography>
          </Tooltip>
          <Tooltip title={filtersVisible ? "Hide Filters" : "Show Filters"}>
            <IconButton 
              size="small" 
              onClick={() => setFiltersVisible(!filtersVisible)}
              color={filtersVisible ? "primary" : "default"}
            >
              <FilterListIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Refresh Data">
            <IconButton 
              size="small" 
              onClick={handleRefresh}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Filter bar */}
      {filtersVisible && (
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 1 }}>
            <Box sx={{ flex: '1 1 250px', minWidth: '200px' }}>
              <TextField
                placeholder="Search models..."
                variant="outlined"
                size="small"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                fullWidth
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Box>

            <Box sx={{ flex: '0 1 150px', minWidth: '120px' }}>
              <FormControl size="small" fullWidth>
                <InputLabel id="stage-select-label">Stage</InputLabel>
                <Select
                  labelId="stage-select-label"
                  value={stageFilter}
                  label="Stage"
                  onChange={(e: SelectChangeEvent) => setStageFilter(e.target.value)}
                >
                  <MenuItem value="all">All Stages</MenuItem>
                  <MenuItem value="production">Production</MenuItem>
                  <MenuItem value="staging">Staging</MenuItem>
                  <MenuItem value="none">None</MenuItem>
                </Select>
              </FormControl>
            </Box>

            {uniqueDevices.length > 0 && (
              <Box sx={{ flex: '0 1 180px', minWidth: '150px' }}>
                <FormControl size="small" fullWidth>
                  <InputLabel id="device-select-label">Device</InputLabel>
                  <Select
                    labelId="device-select-label"
                    value={deviceFilter}
                    label="Device"
                    onChange={(e: SelectChangeEvent) => setDeviceFilter(e.target.value)}
                  >
                    <MenuItem value="all">All Devices</MenuItem>
                    {uniqueDevices.map(device => (
                      <MenuItem key={device} value={device}>{device}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>
            )}
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="outlined"
              size="small"
              onClick={() => {
                setSearchTerm('');
                setStageFilter('all');
                setDeviceFilter('all');
              }}
              disabled={searchTerm === '' && stageFilter === 'all' && deviceFilter === 'all'}
            >
              Reset Filters
            </Button>
          </Box>
        </Box>
      )}
      
      {/* Model table */}
      <TableContainer sx={{ bgcolor: 'transparent' }}>
        <Table sx={{ minWidth: 650 }} size="medium">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'name'}
                  direction={orderBy === 'name' ? order : 'asc'}
                  onClick={() => handleRequestSort('name')}
                >
                  Model
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={orderBy === 'latest_version'}
                  direction={orderBy === 'latest_version' ? order : 'asc'}
                  onClick={() => handleRequestSort('latest_version')}
                >
                  Latest
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">Production</TableCell>
              <TableCell align="center">Staging</TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={orderBy === 'total_versions'}
                  direction={orderBy === 'total_versions' ? order : 'asc'}
                  onClick={() => handleRequestSort('total_versions')}
                >
                  Total Versions
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredModels.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography variant="body1" sx={{ py: 2 }}>No models found matching your filters</Typography>
                </TableCell>
              </TableRow>
            ) : (
              filteredModels
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((model) => (
                  <TableRow 
                    key={model.name}
                    hover
                    onClick={() => handleModelSelect(model.name)}
                    selected={selectedModel === model.name}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>
                      <Box>
                        <Typography variant="body2" fontWeight="medium">{model.name}</Typography>
                        {model.device_id && (
                          <Typography variant="caption" color="text.secondary">{model.device_id}</Typography>
                        )}
                      </Box>
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
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
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
                      </Box>
                    </TableCell>
                  </TableRow>
                ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      {/* Pagination */}
      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={filteredModels.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Box>
  );
};

export default ModelRegistryTable;