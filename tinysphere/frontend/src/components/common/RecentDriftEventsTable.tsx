import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper, 
  Chip, 
  CircularProgress, 
  Typography,
  Button,
  IconButton,
  Tooltip,
  Link
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { 
  VisibilityOutlined as ViewIcon,
  CheckCircleOutline as ValidateIcon,
  WarningAmber as WarningIcon,
  Error as ErrorIcon,
  BubbleChart as DistributionIcon,
  ViewInAr as FeatureIcon,
  RemoveCircleOutline as OutlierIcon,
  QuestionMark as UnknownIcon,
  Psychology as ConfidenceIcon,
  Lightbulb as CustomIcon,
  ReportOutlined as ReportIcon
} from '@mui/icons-material';
import { format, parseISO } from 'date-fns';
import { getDriftEvents } from '../../services/api';
import { DriftEvent, DriftStatus, DriftType } from '../../types/api';
import SectionCard from './SectionCard';

interface RecentDriftEventsTableProps {
  title?: string;
  deviceId?: string;
  limit?: number;
  onViewEvent?: (eventId: string) => void;
}

// Status color mapping
const statusColors: Record<DriftStatus, string> = {
  pending: '#ff9800',
  validated: '#4caf50',
  rejected: '#f44336',
  resolved: '#2196f3',
  ignored: '#9e9e9e'
};

// Status icon mapping
const StatusIcon: React.FC<{ status: DriftStatus }> = ({ status }) => {
  switch (status) {
    case 'validated':
      return <ValidateIcon fontSize="small" />;
    case 'pending':
      return <WarningIcon fontSize="small" />;
    case 'rejected':
      return <ErrorIcon fontSize="small" />;
    default:
      return null;
  }
};

// Drift type icon mapping
const DriftTypeIcon: React.FC<{ type: DriftType }> = ({ type }) => {
  switch (type) {
    case 'confidence':
      return <ConfidenceIcon fontSize="small" />;
    case 'distribution':
      return <DistributionIcon fontSize="small" />;
    case 'feature':
      return <FeatureIcon fontSize="small" />;
    case 'outlier':
      return <OutlierIcon fontSize="small" />;
    case 'custom':
      return <CustomIcon fontSize="small" />;
    default:
      return <UnknownIcon fontSize="small" />;
  }
};

const RecentDriftEventsTable: React.FC<RecentDriftEventsTableProps> = ({
  title = 'Recent Drift Events',
  deviceId,
  limit = 5,
  onViewEvent
}) => {
  const [events, setEvents] = useState<DriftEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoading(true);
        const params: any = { limit };
        
        if (deviceId) {
          params.device_id = deviceId;
        }
        
        const data = await getDriftEvents(params);
        setEvents(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching drift events:', err);
        setError('Failed to load drift events');
      } finally {
        setLoading(false);
      }
    };

    fetchEvents();
  }, [deviceId, limit]);

  const handleViewEvent = (eventId: string) => {
    if (onViewEvent) {
      onViewEvent(eventId);
    }
  };

  if (loading) {
    return (
      <SectionCard 
        title={title}
        icon={<ReportIcon style={{ fontSize: 20, color: '#00647D' }} />}
      >
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      </SectionCard>
    );
  }

  if (error) {
    return (
      <SectionCard 
        title={title}
        icon={<ReportIcon style={{ fontSize: 20, color: '#00647D' }} />}
      >
        <Box sx={{ p: 2 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      </SectionCard>
    );
  }

  if (events.length === 0) {
    return (
      <SectionCard 
        title={title}
        icon={<ReportIcon style={{ fontSize: 20, color: '#00647D' }} />}
      >
        <Box sx={{ p: 2 }}>
          <Typography>No drift events available</Typography>
        </Box>
      </SectionCard>
    );
  }

  return (
    <SectionCard 
      title={title}
      icon={<ReportIcon style={{ fontSize: 20, color: '#00647D' }} />}
    >
      <TableContainer component={Paper} elevation={0}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Event ID</TableCell>
              <TableCell>Device</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Timestamp</TableCell>
              <TableCell align="center">Samples</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {events.map((event) => (
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
                <TableCell align="center">
                  {event.sample_count}
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="View Details">
                    <IconButton 
                      size="small" 
                      onClick={() => handleViewEvent(event.event_id)}
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
      
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
        <Button 
          variant="outlined" 
          component={RouterLink} 
          to="/drift"
          size="small"
        >
          View All Drift Events
        </Button>
      </Box>
    </SectionCard>
  );
};

export default RecentDriftEventsTable;