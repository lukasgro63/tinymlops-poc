import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  TablePagination,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  IconButton,
  Tooltip,
  Alert,
  Link,
  Divider
} from '@mui/material';
import ErrorDisplay from '../components/common/ErrorDisplay';
import { 
  FilterList as FilterIcon,
  Refresh as RefreshIcon,
  WarningAmber as WarningIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Psychology as PsychologyIcon,
  BubbleChart as DistributionIcon,
  ViewInAr as FeatureIcon,
  RemoveCircleOutline as OutlierIcon,
  Help as UnknownIcon,
  VisibilityOutlined as ViewIcon,
  TrendingUp as TrendingUpIcon,
  HistoryEdu as HistoryIcon,
  AssessmentOutlined as AssessmentIcon
} from '@mui/icons-material';
import { format, subDays, parseISO } from 'date-fns';
import { getDriftEvents } from '../services/api';
import { DriftEvent, DriftType, DriftStatus } from '../types/api';
import DriftStatisticsCard from '../components/common/DriftStatisticsCard';

// Status color mapping
const statusColors: Record<DriftStatus, string> = {
  pending: '#ff9800',
  validated: '#4caf50',
  rejected: '#f44336',
  resolved: '#2196f3',
  ignored: '#9e9e9e'
};

// Drift type icon mapping
const DriftTypeIcon: React.FC<{ type: string }> = ({ type }) => {
  switch (type) {
    case 'confidence':
      return <PsychologyIcon fontSize="small" />;
    case 'distribution':
      return <DistributionIcon fontSize="small" />;
    case 'feature':
      return <FeatureIcon fontSize="small" />;
    case 'outlier':
      return <OutlierIcon fontSize="small" />;
    default:
      return <UnknownIcon fontSize="small" />;
  }
};

// Status icon mapping
const StatusIcon: React.FC<{ status: DriftStatus }> = ({ status }) => {
  switch (status) {
    case 'validated':
      return <CheckIcon fontSize="small" />;
    case 'pending':
      return <WarningIcon fontSize="small" />;
    case 'rejected':
      return <CancelIcon fontSize="small" />;
    default:
      return null;
  }
};

const DriftPage: React.FC = () => {
  const navigate = useNavigate();
  
  // Pagination state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  // Data state
  const [events, setEvents] = useState<DriftEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  
  // Filter state
  const [filters, setFilters] = useState<{
    deviceId: string;
    driftType: string;
    status: string;
    startDate: string;
    endDate: string;
  }>({
    deviceId: '',
    driftType: '',
    status: '',
    startDate: format(subDays(new Date(), 30), 'yyyy-MM-dd'),
    endDate: format(new Date(), 'yyyy-MM-dd')
  });
  
  const [filtersVisible, setFiltersVisible] = useState(false);
  
  // Additional state to track statistics loading
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);
  
  // Load drift events
  useEffect(() => {
    fetchEvents();
  }, [page, rowsPerPage, filters]);
  
  const fetchEvents = async () => {
    try {
      setLoading(true);
      
      // Prepare filter parameters
      const params: any = {
        skip: page * rowsPerPage,
        limit: rowsPerPage
      };
      
      if (filters.deviceId) params.device_id = filters.deviceId;
      if (filters.driftType) params.drift_type = filters.driftType;
      if (filters.status) params.status = filters.status;
      if (filters.startDate) params.start_date = new Date(filters.startDate).toISOString();
      if (filters.endDate) params.end_date = new Date(filters.endDate).toISOString();
      
      // Fetch events
      const data = await getDriftEvents(params);
      
      setEvents(data);
      // In a real application, we'd set totalCount from API response headers or metadata
      setTotalCount(data.length > rowsPerPage ? 100 : data.length); // Mock total count
      setError(null);
    } catch (err) {
      console.error('Error fetching drift events:', err);
      setError('Failed to load drift events');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle pagination
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };
  
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  // Handle filter changes
  const handleFilterChange = (field: string, value: string) => {
    setFilters({
      ...filters,
      [field]: value
    });
    setPage(0); // Reset to first page on filter change
  };
  
  const resetFilters = () => {
    setFilters({
      deviceId: '',
      driftType: '',
      status: '',
      startDate: format(subDays(new Date(), 30), 'yyyy-MM-dd'),
      endDate: format(new Date(), 'yyyy-MM-dd')
    });
    setPage(0);
  };
  
  // Handle row click to navigate to event details
  const handleRowClick = (eventId: string) => {
    navigate(`/drift/${eventId}`);
  };

  // Handle page-level loading and error states
  if (loading || error) {
    return (
      <ErrorDisplay 
        error={error}
        loading={loading}
        onRetry={() => fetchEvents()}
        height="70vh"
      />
    );
  }
  
  return (
    <Box sx={{ p: 0 }}>
      
      {/* Statistics dashboard */}
      <Box sx={{ mb: 3 }}>
        <DriftStatisticsCard 
          title="Drift Detection Overview" 
          days={30}
          parentLoading={loading}
          parentError={error}
        />
      </Box>
      
      {/* Events list */}
      <Paper sx={{ mb: 3, p: 2 }} elevation={2}>
        {/* Toolbar */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <HistoryIcon style={{ fontSize: 20, color: '#00647D', marginRight: '8px' }} />
            <Typography variant="h6" component="h2">
              Drift Events
            </Typography>
          </Box>
          
          <Box>
            <Tooltip title="Toggle Filters">
              <IconButton onClick={() => setFiltersVisible(!filtersVisible)}>
                <FilterIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Refresh">
              <IconButton onClick={fetchEvents}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        {/* Filters */}
        {filtersVisible && (
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
                <TextField
                  label="Device ID"
                  value={filters.deviceId}
                  onChange={(e) => handleFilterChange('deviceId', e.target.value)}
                  fullWidth
                  size="small"
                />
              </Box>
              
              <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
                <FormControl fullWidth size="small">
                  <InputLabel id="drift-type-select-label">Drift Type</InputLabel>
                  <Select
                    labelId="drift-type-select-label"
                    value={filters.driftType}
                    label="Drift Type"
                    onChange={(e) => handleFilterChange('driftType', e.target.value)}
                  >
                    <MenuItem value="">All</MenuItem>
                    <MenuItem value="confidence">Confidence</MenuItem>
                    <MenuItem value="distribution">Distribution</MenuItem>
                    <MenuItem value="feature">Feature</MenuItem>
                    <MenuItem value="outlier">Outlier</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              
              <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
                <FormControl fullWidth size="small">
                  <InputLabel id="status-select-label">Status</InputLabel>
                  <Select
                    labelId="status-select-label"
                    value={filters.status}
                    label="Status"
                    onChange={(e) => handleFilterChange('status', e.target.value)}
                  >
                    <MenuItem value="">All</MenuItem>
                    <MenuItem value="pending">Pending</MenuItem>
                    <MenuItem value="validated">Validated</MenuItem>
                    <MenuItem value="rejected">Rejected</MenuItem>
                    <MenuItem value="resolved">Resolved</MenuItem>
                    <MenuItem value="ignored">Ignored</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              
              <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
                <TextField
                  label="Start Date"
                  type="date"
                  value={filters.startDate}
                  onChange={(e) => handleFilterChange('startDate', e.target.value)}
                  fullWidth
                  size="small"
                  InputLabelProps={{ shrink: true }}
                />
              </Box>
              
              <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
                <TextField
                  label="End Date"
                  type="date"
                  value={filters.endDate}
                  onChange={(e) => handleFilterChange('endDate', e.target.value)}
                  fullWidth
                  size="small"
                  InputLabelProps={{ shrink: true }}
                />
              </Box>
              
              <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
                <Button 
                  variant="outlined" 
                  onClick={resetFilters}
                  fullWidth
                >
                  Reset Filters
                </Button>
              </Box>
            </Box>
            
            <Divider sx={{ mt: 2 }} />
          </Box>
        )}
        
        {/* Empty state */}
        {events.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary">
              No drift events found
            </Typography>
            <Button
              variant="outlined"
              onClick={resetFilters}
              sx={{ mt: 2 }}
            >
              Reset Filters
            </Button>
          </Box>
        ) : (
          <>
            {/* Events table */}
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Event ID</TableCell>
                    <TableCell>Device</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Detected At</TableCell>
                    <TableCell>Samples</TableCell>
                    <TableCell>Validations</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {events.map((event) => (
                    <TableRow 
                      key={event.event_id}
                      hover
                      onClick={() => handleRowClick(event.event_id)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Tooltip title={event.event_id}>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {event.event_id.substring(0, 8)}...
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Link 
                          href={`/devices/${event.device_id}`}
                          underline="hover"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {event.device_id}
                        </Link>
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={<DriftTypeIcon type={event.drift_type} />}
                          label={event.drift_type.charAt(0).toUpperCase() + event.drift_type.slice(1)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={<StatusIcon status={event.status} />}
                          label={event.status.charAt(0).toUpperCase() + event.status.slice(1)}
                          size="small"
                          sx={{ 
                            backgroundColor: `${statusColors[event.status]}20`,
                            color: statusColors[event.status],
                            borderColor: statusColors[event.status]
                          }}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        {event.timestamp && format(parseISO(event.timestamp), 'MMM d, yyyy HH:mm')}
                      </TableCell>
                      <TableCell>{event.sample_count}</TableCell>
                      <TableCell>{event.validation_count}</TableCell>
                      <TableCell align="right">
                        <IconButton 
                          size="small" 
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRowClick(event.event_id);
                          }}
                        >
                          <ViewIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            {/* Pagination */}
            <TablePagination
              rowsPerPageOptions={[5, 10, 25, 50]}
              component="div"
              count={totalCount}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </>
        )}
      </Paper>
    </Box>
  );
};

export default DriftPage;