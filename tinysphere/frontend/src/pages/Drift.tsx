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
  AssessmentOutlined as AssessmentIcon,
  Sort as SortIcon,
  ArrowUpward as SortAscIcon,
  ArrowDownward as SortDescIcon
} from '@mui/icons-material';
import { format, subDays, parseISO } from 'date-fns';
import { 
  getDriftEvents,
  getDriftStatistics,
  getDriftStatisticsByDevice,
  getDriftTimelineByDevice,
  DeviceStatusData,
  DriftTimelineData
} from '../services/api';
import { DriftEvent, DriftType, DriftStatus } from '../types/api';
import DriftStatisticsCard from '../components/common/DriftStatisticsCard';
import DriftStatusByDeviceChart from '../components/common/DriftStatusByDeviceChart';
import DriftEventsByDeviceChart from '../components/common/DriftEventsByDeviceChart';
import SectionCard from '../components/common/SectionCard';

// Helper function to get MUI color based on status
const getStatusColor = (status: DriftStatus): 'success' | 'warning' | 'error' | 'info' | 'default' => {
  switch (status) {
    case 'validated':
      return 'success';
    case 'pending':
      return 'warning';
    case 'rejected':
      return 'error';
    case 'resolved':
      return 'info';
    case 'ignored':
      return 'default';
  }
};

// Drift type icon mapping
const DriftTypeIcon: React.FC<{ type: string }> = ({ type }) => {
  // Normalize the drift type to lowercase for consistent handling
  const normalizedType = type.toLowerCase();
  
  switch (normalizedType) {
    case 'confidence':
      return <PsychologyIcon fontSize="small" />;
    case 'distribution':
      return <DistributionIcon fontSize="small" />;
    case 'feature':
      return <FeatureIcon fontSize="small" />;
    case 'outlier':
      return <OutlierIcon fontSize="small" />;
    case 'knn_distance':
      return <TrendingUpIcon fontSize="small" />;
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
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc'); // desc = newest first (default)
  
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
    // Set end date to tomorrow to account for timezone differences
    endDate: format(new Date(new Date().getTime() + 24 * 60 * 60 * 1000), 'yyyy-MM-dd')
  });
  
  const [filtersVisible, setFiltersVisible] = useState(true); // Default to visible
  
  // State for charts
  const [deviceStatusData, setDeviceStatusData] = useState<DeviceStatusData[]>([]);
  const [deviceTimelineData, setDeviceTimelineData] = useState<DriftTimelineData[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(true);
  const [timelineError, setTimelineError] = useState<string | null>(null);
  const [statusChartLoading, setStatusChartLoading] = useState(true);
  const [statusChartError, setStatusChartError] = useState<string | null>(null);
  const [timelineRange, setTimelineRange] = useState<'day' | 'week' | 'month'>('week');
  
  // Load initial data when component mounts
  useEffect(() => {
    console.log("Drift Page: Loading initial data");
    fetchEvents();
    fetchDeviceStatusData();
    fetchDeviceTimelineData();
    // Auto-refresh removed as per requirement
  }, [page, rowsPerPage]);

  // Re-fetch when filters or sort order change
  useEffect(() => {
    console.log("Drift Page: Filters or sort order changed, re-fetching");
    fetchEvents();
  }, [filters, sortOrder]);
  
  // Re-fetch timeline when timeline range changes
  useEffect(() => {
    fetchDeviceTimelineData();
  }, [timelineRange, filters.driftType]);
  
  const fetchEvents = async () => {
    try {
      setLoading(true);
      
      // Prepare filter parameters
      const params: any = {
        skip: page * rowsPerPage,
        limit: rowsPerPage,
        sort_order: sortOrder // Übergebe Sortierreihenfolge an API
      };
      
      if (filters.deviceId) params.device_id = filters.deviceId;
      if (filters.driftType) params.drift_type = filters.driftType.toLowerCase(); // Ensure lowercase for drift_type
      if (filters.status) params.status = filters.status;
      if (filters.startDate) {
        // For start date, set to beginning of the day in local timezone
        const startDate = new Date(filters.startDate);
        startDate.setHours(0, 0, 0, 0);
        params.start_date = startDate.toISOString();
      }

      if (filters.endDate) {
        // For end date, set to end of the day in local timezone
        const endDate = new Date(filters.endDate);
        endDate.setHours(23, 59, 59, 999);
        params.end_date = endDate.toISOString();
      }
      
      // Fetch events
      try {
        const data = await getDriftEvents(params);

        if (Array.isArray(data)) {
          console.log("Successfully fetched drift events:", data.length);
          
          // The sorting is now done on the server level with the sort_order parameter
          setEvents(data);
          // In a real application, we'd set totalCount from API response headers or metadata
          setTotalCount(data.length > rowsPerPage ? 100 : data.length); // Mock total count
        } else {
          console.error("Unexpected response format:", data);
          setEvents([]);
          setTotalCount(0);
        }
      } catch (fetchErr) {
        console.error("Error in getDriftEvents:", fetchErr);
        setEvents([]);
        setTotalCount(0);
      }
      setError(null);
    } catch (err) {
      console.error('Error fetching drift events:', err);
      setError('Failed to load drift events');
    } finally {
      setLoading(false);
    }
  };
  
  const fetchDeviceStatusData = async () => {
    try {
      setStatusChartLoading(true);
      const days = 30; // Use a fixed 30-day window for status data
      
      const driftType = filters.driftType || undefined;
      const data = await getDriftStatisticsByDevice(days, driftType);
      setDeviceStatusData(data);
      setStatusChartError(null);
    } catch (err) {
      console.error('Error fetching device status data:', err);
      setStatusChartError('Failed to load device status data');
    } finally {
      setStatusChartLoading(false);
    }
  };
  
  const fetchDeviceTimelineData = async () => {
    try {
      setTimelineLoading(true);
      
      const driftType = filters.driftType || undefined;
      const data = await getDriftTimelineByDevice(timelineRange, driftType);
      setDeviceTimelineData(data);
      setTimelineError(null);
    } catch (err) {
      console.error('Error fetching device timeline data:', err);
      setTimelineError('Failed to load device timeline data');
    } finally {
      setTimelineLoading(false);
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
    
    // Refresh status chart when drift type changes
    if (field === 'driftType') {
      fetchDeviceStatusData();
    }
  };
  
  const resetFilters = () => {
    setFilters({
      deviceId: '',
      driftType: '',
      status: '',
      startDate: format(subDays(new Date(), 30), 'yyyy-MM-dd'),
      // Set end date to tomorrow to account for timezone differences
      endDate: format(new Date(new Date().getTime() + 24 * 60 * 60 * 1000), 'yyyy-MM-dd')
    });
    setPage(0);
    
    // Refresh charts
    fetchDeviceStatusData();
    fetchDeviceTimelineData();
  };
  
  // Handle row click to navigate to event details
  const handleRowClick = (eventId: string) => {
    navigate(`/drift/${eventId}`);
  };
  
  // Toggle sort order
  const handleToggleSortOrder = () => {
    setSortOrder(prevOrder => prevOrder === 'asc' ? 'desc' : 'asc');
    setPage(0); // Reset to first page when sorting changes
  };
  
  // Handle timeline range change
  const handleTimelineRangeChange = (range: 'day' | 'week' | 'month') => {
    setTimelineRange(range);
  };
  
  return (
    <Box sx={{ p: 0 }}>
      {/* Header and Statistics Dashboard */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* First row: Charts side by side */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, maxHeight: '450px' }}>
          {/* Left chart: Device Status Chart */}
          <Box sx={{ flex: '1 1 400px', minWidth: '300px' }}>
            <SectionCard 
              title="Drift Status by Device" 
              icon={<AssessmentIcon style={{ fontSize: 20, color: '#00647D' }} />}
              height={400}
            >
              <DriftStatusByDeviceChart
                data={deviceStatusData}
                isLoading={statusChartLoading}
                error={statusChartError}
                onRetry={fetchDeviceStatusData}
                height={350}
              />
            </SectionCard>
          </Box>
          
          {/* Right chart: Drift Events By Device Chart */}
          <Box sx={{ flex: '1 1 400px', minWidth: '300px' }}>
            <SectionCard 
              title="Drift Events by Device" 
              icon={<AssessmentIcon style={{ fontSize: 20, color: '#00647D' }} />}
              height={400}
            >
              <DriftEventsByDeviceChart
                data={deviceTimelineData}
                isLoading={timelineLoading}
                error={timelineError}
                onRetry={fetchDeviceTimelineData}
                currentTimeRange={timelineRange}
                onTimeRangeChange={handleTimelineRangeChange}
                height={350}
              />
            </SectionCard>
          </Box>
        </Box>
        
        {/* Events list */}
        <Box>
          <Paper sx={{ p: 2 }} elevation={2}>
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
                  <IconButton onClick={() => {
                    fetchEvents();
                    fetchDeviceStatusData();
                    fetchDeviceTimelineData();
                  }}>
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
                        <MenuItem value="knn_distance">KNN Distance</MenuItem>
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
                  
                  <Box sx={{ flex: '1 1 200px', minWidth: '150px', display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Button 
                      variant="outlined" 
                      onClick={resetFilters}
                      sx={{ flexGrow: 1 }}
                    >
                      Reset Filters
                    </Button>
                    <Tooltip title={sortOrder === 'desc' ? "Showing newest first - Click to show oldest first" : "Showing oldest first - Click to show newest first"}>
                      <IconButton onClick={handleToggleSortOrder} color="primary">
                        {sortOrder === 'desc' ? <SortDescIcon /> : <SortAscIcon />}
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                
                <Divider sx={{ mt: 2 }} />
              </Box>
            )}
            
            {/* Loading or error states */}
            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <Typography>Loading...</Typography>
              </Box>
            )}
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            
            {/* Empty state */}
            {!loading && !error && events.length === 0 ? (
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
                              label={event.drift_type.split('_')
                                .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                                .join(' ')}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={event.status.charAt(0).toUpperCase() + event.status.slice(1)}
                              size="small"
                              color={getStatusColor(event.status)}
                              sx={getStatusColor(event.status) === 'default' ? { backgroundColor: '#FFA500' } : {}}
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
      </Box>
    </Box>
  );
};

export default DriftPage;