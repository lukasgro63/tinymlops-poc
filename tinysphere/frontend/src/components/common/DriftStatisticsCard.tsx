import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Link,
  Divider
} from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import { 
  AssessmentOutlined as AssessmentIcon,
  VisibilityOutlined as ViewIcon,
  Psychology as PsychologyIcon,
  BubbleChart as DistributionIcon,
  ViewInAr as FeatureIcon,
  RemoveCircleOutline as OutlierIcon,
  Help as UnknownIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';
import { format, parseISO } from 'date-fns';
import { getDriftStatistics } from '../../services/api';
import { DriftStatistics, DriftStatus, DriftType } from '../../types/api';
import SectionCard from './SectionCard';
import ErrorDisplay from './ErrorDisplay';

interface DriftStatisticsCardProps {
  title?: string;
  deviceId?: string;
  days?: number;
  parentLoading?: boolean;
  parentError?: string | null;
  showRecentEvents?: boolean; // Control whether to show recent events table
}

// Status color mapping
const statusColors: Record<DriftStatus, string> = {
  pending: '#ff9800',
  validated: '#4caf50',
  rejected: '#f44336',
  resolved: '#2196f3',
  ignored: '#9e9e9e'
};

// Type color mapping
const typeColors: Record<string, string> = {
  confidence: '#8884d8',
  distribution: '#82ca9d',
  feature: '#ffc658',
  outlier: '#ff8042',
  custom: '#a4de6c',
  knn_distance: '#6495ED', // Add color for knn_distance
  unknown: '#d0d0d0'
};

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
  switch (type) {
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

const DriftStatisticsCard: React.FC<DriftStatisticsCardProps> = ({ 
  title = 'Drift Detection Statistics', 
  deviceId,
  days = 30,
  parentLoading,
  parentError,
  showRecentEvents = false // Default to not showing recent events
}) => {
  const [statistics, setStatistics] = useState<DriftStatistics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Skip loading if parent has an error or is loading
    if (parentLoading || parentError) {
      return;
    }
    
    const fetchStatistics = async () => {
      try {
        setLoading(true);
        const data = await getDriftStatistics(deviceId, days);
        setStatistics(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching drift statistics:', err);
        setError('Failed to load drift statistics');
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, [deviceId, days, parentLoading, parentError]);


  // If parent has errors, don't show our own error state
  if ((loading && !parentLoading) || (error && !parentError)) {
    return (
      <SectionCard 
        title={title}
        icon={<AssessmentIcon style={{ fontSize: 20, color: '#00647D' }} />}
      >
        <ErrorDisplay
          error={error}
          loading={loading}
          onRetry={() => getDriftStatistics(deviceId, days).then(setStatistics).catch(e => setError(e.message))}
          height="300px"
        />
      </SectionCard>
    );
  }

  if (!statistics) {
    return (
      <SectionCard 
        title={title}
        icon={<AssessmentIcon style={{ fontSize: 20, color: '#00647D' }} />}
      >
        <Box sx={{ p: 2 }}>
          <Typography>No drift statistics available</Typography>
        </Box>
      </SectionCard>
    );
  }

  // Prepare data for status distribution chart
  const statusData = Object.entries(statistics.by_status).map(([status, count]) => ({
    name: status.charAt(0).toUpperCase() + status.slice(1),
    count,
    color: statusColors[status as DriftStatus] || '#d0d0d0'
  }));

  // Prepare data for type distribution chart
  const typeData = Object.entries(statistics.by_type).map(([type, count]) => ({
    name: type.charAt(0).toUpperCase() + type.slice(1),
    count,
    color: typeColors[type] || '#d0d0d0'
  }));

  return (
    <SectionCard 
      title={title} 
      icon={<AssessmentIcon style={{ fontSize: 20, color: '#00647D' }} />}
    >
      <CardContent>
        <div style={{ display: 'grid', gap: '24px' }}>
          {/* Summary statistics */}
          <div>
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6" component="div">
                Total Drift Events: {statistics.total_events}
              </Typography>
              <Box>
                {Object.entries(statistics.by_status).map(([status, count]) => (
                  <Chip 
                    key={status}
                    label={`${status.charAt(0).toUpperCase() + status.slice(1)}: ${count}`}
                    sx={{ 
                      m: 0.5, 
                      bgcolor: statusColors[status as DriftStatus] || '#d0d0d0',
                      color: '#fff'
                    }}
                    size="small"
                  />
                ))}
              </Box>
            </Box>
          </div>

          {/* Time series chart */}
          <div>
            <Typography variant="subtitle1" gutterBottom>
              Drift Events Over Time
            </Typography>
            <Box sx={{ width: '100%', height: 200 }}>
              <ResponsiveContainer>
                <BarChart
                  data={statistics.by_day}
                  margin={{ top: 10, right: 10, left: 0, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    angle={-45} 
                    textAnchor="end"
                    tick={{ fontSize: 12 }}
                    height={50}
                  />
                  <YAxis />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#0b2a5a" name="Events" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </div>

          {/* Status and Type distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <div>
              <Typography variant="subtitle1" gutterBottom>
                Status Distribution
              </Typography>
              <Box sx={{ width: '100%', height: 200 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={statusData}
                    margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis 
                      dataKey="name" 
                      type="category" 
                      width={80}
                      tick={{ fontSize: 12 }}
                    />
                    <RechartsTooltip />
                    <Bar 
                      dataKey="count" 
                      name="Events" 
                      fill="#0b2a5a"
                      radius={[0, 4, 4, 0]}
                    >
                      {statusData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color || '#0b2a5a'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </div>

            <div>
              <Typography variant="subtitle1" gutterBottom>
                Drift Type Distribution
              </Typography>
              <Box sx={{ width: '100%', height: 200 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={typeData}
                    margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis 
                      dataKey="name" 
                      type="category"
                      width={80}
                      tick={{ fontSize: 12 }}
                    />
                    <RechartsTooltip />
                    <Bar 
                      dataKey="count" 
                      name="Events" 
                      fill="#0b2a5a"
                      radius={[0, 4, 4, 0]}
                    >
                      {typeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color || '#0b2a5a'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </div>
          </div>
          
          {/* Recent Drift Events Table - Only shown if showRecentEvents is true */}
          {showRecentEvents && statistics.recent_events && statistics.recent_events.length > 0 && (
            <div>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle1" gutterBottom>
                Recent Drift Events
              </Typography>
              <TableContainer component={Paper} elevation={0} sx={{ maxHeight: 300 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Event ID</TableCell>
                      <TableCell>Device</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Timestamp</TableCell>
                      <TableCell align="center">Samples</TableCell>
                      <TableCell align="center">Validations</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {statistics.recent_events.map((event) => (
                      <TableRow key={event.event_id}>
                        <TableCell>
                          <Tooltip title={event.event_id}>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {event.event_id.substring(0, 8)}...
                            </Typography>
                          </Tooltip>
                        </TableCell>
                        <TableCell>
                          <Link 
                            component={RouterLink} 
                            to={`/devices/${event.device_id}`}
                            underline="hover"
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
                            label={event.status.charAt(0).toUpperCase() + event.status.slice(1)}
                            size="small"
                            color={getStatusColor(event.status)}
                            sx={getStatusColor(event.status) === 'default' ? { backgroundColor: '#FFA500' } : {}}
                          />
                        </TableCell>
                        <TableCell>
                          {event.timestamp && format(parseISO(event.timestamp), 'MMM d, yyyy HH:mm')}
                        </TableCell>
                        <TableCell align="center">
                          {event.sample_count}
                        </TableCell>
                        <TableCell align="center">
                          {event.validation_count}
                        </TableCell>
                        <TableCell align="right">
                          <Tooltip title="View Details">
                            <IconButton 
                              size="small" 
                              component={RouterLink}
                              to={`/drift/${event.event_id}`}
                            >
                              <ViewIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </div>
          )}
        </div>
      </CardContent>
    </SectionCard>
  );
};

export default DriftStatisticsCard;