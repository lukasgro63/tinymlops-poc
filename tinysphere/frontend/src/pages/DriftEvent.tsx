import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Grid, 
  Paper, 
  Chip, 
  Button, 
  CircularProgress, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider,
  Link,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Switch,
  Stack,
  Alert,
  IconButton,
  Tooltip,
  LinearProgress,
  MenuItem,
  Select,
  FormControl,
  InputLabel
} from '@mui/material';
import ErrorDisplay from '../components/common/ErrorDisplay';
import { 
  WarningAmber as WarningIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Psychology as PsychologyIcon,
  BubbleChart as DistributionIcon,
  ViewInAr as FeatureIcon,
  RemoveCircleOutline as OutlierIcon,
  Help as UnknownIcon,
  Image as ImageIcon,
  DynamicFeed as VectorIcon
} from '@mui/icons-material';
import { format, parseISO } from 'date-fns';
import { 
  getDriftEvent, 
  getDriftSamples, 
  validateDriftSample,
  updateDriftEventStatus
} from '../services/api';
import { DriftEvent as DriftEventType, DriftSample, DriftStatus } from '../types/api';

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

// Status options
const statusOptions = [
  { value: 'pending', label: 'Pending' },
  { value: 'validated', label: 'Validated' },
  { value: 'rejected', label: 'Rejected' },
  { value: 'resolved', label: 'Resolved' },
  { value: 'ignored', label: 'Ignored' }
];

// Drift type icon mapping
const DriftTypeIcon: React.FC<{ type: string }> = ({ type }) => {
  switch (type) {
    case 'confidence':
      return <PsychologyIcon />;
    case 'distribution':
      return <DistributionIcon />;
    case 'feature':
      return <FeatureIcon />;
    case 'outlier':
      return <OutlierIcon />;
    default:
      return <UnknownIcon />;
  }
};

interface ValidationDialogProps {
  open: boolean;
  sample: DriftSample | null;
  onClose: () => void;
  onValidate: (validation: {
    is_valid_drift: boolean;
    true_label?: string;
    notes?: string;
  }) => void;
}

const ValidationDialog: React.FC<ValidationDialogProps> = ({
  open,
  sample,
  onClose,
  onValidate
}) => {
  const [isValid, setIsValid] = useState(true);
  const [trueLabel, setTrueLabel] = useState('');
  const [notes, setNotes] = useState('');

  useEffect(() => {
    if (sample) {
      setIsValid(true);
      setTrueLabel(sample.prediction || '');
      setNotes('');
    }
  }, [sample]);

  const handleSubmit = () => {
    onValidate({
      is_valid_drift: isValid,
      true_label: trueLabel,
      notes
    });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Validate Drift Sample</DialogTitle>
      <DialogContent>
        {sample && (
          <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Sample ID: {sample.sample_id}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Prediction: {sample.prediction || 'None'}
              </Typography>
              {sample.confidence && (
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Confidence: {(sample.confidence * 100).toFixed(2)}%
                </Typography>
              )}
            </Box>

            <Box>
              <FormControlLabel
                control={
                  <Switch
                    checked={isValid}
                    onChange={(e) => setIsValid(e.target.checked)}
                    color="primary"
                  />
                }
                label="This is a valid drift case"
              />
            </Box>

            <Box>
              <TextField
                label="True Label"
                value={trueLabel}
                onChange={(e) => setTrueLabel(e.target.value)}
                fullWidth
                variant="outlined"
                disabled={!isValid}
                helperText={isValid ? "Enter the correct label for this sample" : "No label needed for invalid drift"}
                size="small"
              />
            </Box>

            <Box>
              <TextField
                label="Notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                fullWidth
                variant="outlined"
                multiline
                rows={3}
                placeholder="Add any notes about this validation..."
                size="small"
              />
            </Box>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          onClick={handleSubmit} 
          variant="contained" 
          color="primary"
        >
          Submit Validation
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const StatusUpdateDialog: React.FC<{
  open: boolean;
  currentStatus: DriftStatus;
  onClose: () => void;
  onUpdate: (status: DriftStatus, notes?: string) => void;
}> = ({ open, currentStatus, onClose, onUpdate }) => {
  const [status, setStatus] = useState<DriftStatus>(currentStatus);
  const [notes, setNotes] = useState('');

  useEffect(() => {
    if (open) {
      setStatus(currentStatus);
      setNotes('');
    }
  }, [open, currentStatus]);

  const handleSubmit = () => {
    onUpdate(status, notes);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Update Drift Event Status</DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box>
              <FormControl fullWidth size="small">
                <InputLabel id="status-select-label">Status</InputLabel>
                <Select
                  labelId="status-select-label"
                  value={status}
                  label="Status"
                  onChange={(e) => setStatus(e.target.value as DriftStatus)}
                >
                  {statusOptions.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Chip
                          label=""
                          size="small"
                          color={getStatusColor(option.value as DriftStatus)}
                          sx={{
                            width: 16,
                            height: 16,
                            mr: 1,
                            '& .MuiChip-label': { padding: 0 }
                          }}
                        />
                        {option.label}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            <Box>
              <TextField
                label="Resolution Notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                fullWidth
                variant="outlined"
                multiline
                rows={3}
                placeholder="Add any notes about this status update..."
                size="small"
              />
            </Box>
          </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          onClick={handleSubmit} 
          variant="contained" 
          color="primary"
        >
          Update Status
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const DriftEventPage: React.FC = () => {
  const { eventId } = useParams<{ eventId: string }>();
  const navigate = useNavigate();
  
  const [event, setEvent] = useState<DriftEventType | null>(null);
  const [samples, setSamples] = useState<DriftSample[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  const [validationDialogOpen, setValidationDialogOpen] = useState(false);
  const [selectedSample, setSelectedSample] = useState<DriftSample | null>(null);
  const [validationLoading, setValidationLoading] = useState(false);
  
  const [statusDialogOpen, setStatusDialogOpen] = useState(false);
  const [statusUpdateLoading, setStatusUpdateLoading] = useState(false);
  const [statusUpdateSuccess, setStatusUpdateSuccess] = useState(false);

  // Fetch event and samples data
  useEffect(() => {
    const fetchEventData = async () => {
      if (!eventId) return;

      try {
        setLoading(true);

        // Fetch event details first
        const eventData = await getDriftEvent(eventId);
        setEvent(eventData);

        try {
          // Then try to get samples
          const samplesData = await getDriftSamples(eventId);
          setSamples(samplesData);
        } catch (sampleErr) {
          console.error('Error fetching samples:', sampleErr);
          // If samples fail, it's not fatal - we can still show event details
          setSamples([]);
        }

        setError(null);
      } catch (err) {
        console.error('Error fetching drift event data:', err);
        setError('Failed to load drift event data. The event may not exist or might have been deleted.');
      } finally {
        setLoading(false);
      }
    };

    fetchEventData();
  }, [eventId]);

  // Handle sample validation
  const handleOpenValidationDialog = (sample: DriftSample) => {
    setSelectedSample(sample);
    setValidationDialogOpen(true);
  };

  const handleCloseValidationDialog = () => {
    setValidationDialogOpen(false);
    setSelectedSample(null);
  };

  const handleValidateSample = async (validation: {
    is_valid_drift: boolean;
    true_label?: string;
    notes?: string;
  }) => {
    if (!selectedSample) return;
    
    try {
      setValidationLoading(true);
      
      // Add validator information
      const validationData = {
        ...validation,
        validated_by: 'UI User' // Could be replaced with actual user info
      };
      
      // Submit validation
      const result = await validateDriftSample(selectedSample.sample_id, validationData);
      
      // Update local data
      setSamples(samples.map(sample => 
        sample.sample_id === selectedSample.sample_id
          ? {
              ...sample,
              status: 'validated',
              true_label: validation.true_label
            }
          : sample
      ));
      
      // Close dialog
      handleCloseValidationDialog();
      
      // Show success message
      // You could add a snackbar or alert here
      
    } catch (err) {
      console.error('Error validating sample:', err);
      // Show error message
      // You could add a snackbar or alert here
    } finally {
      setValidationLoading(false);
    }
  };

  // Handle status update
  const handleOpenStatusDialog = () => {
    if (!event) return;
    setStatusDialogOpen(true);
  };

  const handleCloseStatusDialog = () => {
    setStatusDialogOpen(false);
  };

  const handleUpdateStatus = async (status: DriftStatus, notes?: string) => {
    if (!event || !eventId) return;
    
    try {
      setStatusUpdateLoading(true);
      
      // Submit status update
      const updatedEvent = await updateDriftEventStatus(eventId, status, notes);
      
      // Update local data
      setEvent(updatedEvent);
      
      // Close dialog
      handleCloseStatusDialog();
      
      // Show success message
      setStatusUpdateSuccess(true);
      setTimeout(() => setStatusUpdateSuccess(false), 5000);
      
    } catch (err) {
      console.error('Error updating event status:', err);
      // Show error message
      // You could add a snackbar or alert here
    } finally {
      setStatusUpdateLoading(false);
    }
  };

  // Loading or error state
  if (loading || error) {
    return (
      <ErrorDisplay 
        error={error}
        loading={loading}
        onRetry={() => {
          if (eventId) {
            setLoading(true);
            setError(null);

            // Fetch event first, then samples
            getDriftEvent(eventId)
              .then(eventData => {
                setEvent(eventData);

                // Then try to get samples
                return getDriftSamples(eventId)
                  .then(samplesData => {
                    setSamples(samplesData);
                  })
                  .catch(sampleErr => {
                    console.error('Error fetching samples:', sampleErr);
                    // If samples fail, it's not fatal
                    setSamples([]);
                  });
              })
              .catch(err => {
                console.error('Error fetching drift event data:', err);
                setError('Failed to load drift event data. The event may not exist or might have been deleted.');
              })
              .finally(() => {
                setLoading(false);
              });
          }
        }}
        height="70vh"
      />
    );
  }

  if (!event) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">Drift event not found or could not be loaded</Alert>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 2 }}>
          The event might have been deleted or there could be an issue with the server. If this error persists, please contact your administrator.
        </Typography>
        <Button
          variant="outlined"
          onClick={() => navigate('/drift')}
          sx={{ mt: 2 }}
        >
          Back to Drift Events
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 0 }}>
      {statusUpdateLoading && <LinearProgress />}
      
      {statusUpdateSuccess && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Event status updated successfully
        </Alert>
      )}
      
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Drift Event Details
        </Typography>
        
        <Box>
          <Button 
            variant="outlined" 
            onClick={() => navigate('/drift')}
            sx={{ mr: 1 }}
          >
            Back to All Events
          </Button>
          
          <Button 
            variant="contained" 
            onClick={handleOpenStatusDialog}
            color="primary"
          >
            Update Status
          </Button>
        </Box>
      </Box>
      
      {/* Event summary */}
      <Paper sx={{ p: 3, mb: 3 }} elevation={2}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
          <Box sx={{ flex: '1 1 500px', minWidth: '300px' }}>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
              <DriftTypeIcon type={event.drift_type} />
              <Typography variant="h6" component="h2">
                {event.drift_type.charAt(0).toUpperCase() + event.drift_type.slice(1)} Drift
              </Typography>
              <Chip
                label={event.status.charAt(0).toUpperCase() + event.status.slice(1)}
                size="small"
                color={getStatusColor(event.status)}
                sx={getStatusColor(event.status) === 'default' ? { backgroundColor: '#FFA500' } : {}}
              />
            </Stack>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ flex: '1 1 150px', minWidth: '120px' }}>
                <Typography variant="body2" color="text.secondary">
                  Event ID
                </Typography>
                <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                  {event.event_id}
                </Typography>
              </Box>
              
              <Box sx={{ flex: '1 1 150px', minWidth: '120px' }}>
                <Typography variant="body2" color="text.secondary">
                  Device ID
                </Typography>
                <Typography variant="body1">
                  <Link href={`/devices/${event.device_id}`} underline="hover">
                    {event.device_id}
                  </Link>
                </Typography>
              </Box>
              
              <Box sx={{ flex: '1 1 150px', minWidth: '120px' }}>
                <Typography variant="body2" color="text.secondary">
                  Model ID
                </Typography>
                <Typography variant="body1">
                  {event.model_id || 'N/A'}
                </Typography>
              </Box>
              
              <Box sx={{ flex: '1 1 150px', minWidth: '120px' }}>
                <Typography variant="body2" color="text.secondary">
                  Detector
                </Typography>
                <Typography variant="body1">
                  {event.detector_name || 'Unknown'}
                </Typography>
              </Box>
              
              <Box sx={{ flex: '1 1 150px', minWidth: '120px' }}>
                <Typography variant="body2" color="text.secondary">
                  Drift Score
                </Typography>
                <Typography variant="body1">
                  {event.drift_score !== undefined ? event.drift_score.toFixed(4) : 'N/A'}
                </Typography>
              </Box>
              
              <Box sx={{ flex: '1 1 150px', minWidth: '120px' }}>
                <Typography variant="body2" color="text.secondary">
                  Detected At
                </Typography>
                <Typography variant="body1">
                  {event.timestamp && format(parseISO(event.timestamp), 'MMM d, yyyy HH:mm:ss')}
                </Typography>
              </Box>
            </Box>
          </Box>
          
          <Box sx={{ flex: '1 1 300px', minWidth: '250px' }}>
            <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }} variant="outlined">
              <Typography variant="subtitle2" gutterBottom>
                Event Statistics
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', my: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Total Samples
                </Typography>
                <Typography variant="body1" fontWeight="medium">
                  {event.sample_count}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', my: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Validated Samples
                </Typography>
                <Typography variant="body1" fontWeight="medium">
                  {samples.filter(s => s.status === 'validated').length}
                </Typography>
              </Box>
              
              <Divider sx={{ my: 1.5 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Resolution
              </Typography>
              
              {event.resolved_at ? (
                <>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', my: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Resolved At
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {format(parseISO(event.resolved_at), 'MMM d, yyyy')}
                    </Typography>
                  </Box>
                  
                  {event.resolution_notes && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Notes
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 0.5 }}>
                        {event.resolution_notes}
                      </Typography>
                    </Box>
                  )}
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Not yet resolved
                </Typography>
              )}
            </Paper>
          </Box>
        </Box>
      </Paper>
      
      {/* Samples table */}
      <Card sx={{ mb: 3 }}>
        <CardHeader 
          title="Drift Samples" 
          titleTypographyProps={{ variant: 'h6' }}
          action={
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                {samples.length} samples
              </Typography>
            </Box>
          }
        />
        
        <Divider />
        
        {samples.length === 0 ? (
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <Typography color="text.secondary">
                No samples available for this drift event
              </Typography>
            </Box>
          </CardContent>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Sample ID</TableCell>
                  <TableCell>Prediction</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Drift Score</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>True Label</TableCell>
                  <TableCell align="right">Data</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {samples.map((sample) => (
                  <TableRow key={sample.sample_id}>
                    <TableCell>
                      <Tooltip title={sample.sample_id}>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {sample.sample_id.substring(0, 8)}...
                        </Typography>
                      </Tooltip>
                    </TableCell>
                    <TableCell>{sample.prediction || 'N/A'}</TableCell>
                    <TableCell>
                      {sample.confidence !== undefined 
                        ? `${(sample.confidence * 100).toFixed(2)}%`
                        : 'N/A'
                      }
                    </TableCell>
                    <TableCell>
                      {sample.drift_score !== undefined 
                        ? sample.drift_score.toFixed(4)
                        : 'N/A'
                      }
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={sample.status.charAt(0).toUpperCase() + sample.status.slice(1)}
                        size="small"
                        color={getStatusColor(sample.status as DriftStatus)}
                        sx={getStatusColor(sample.status as DriftStatus) === 'default' ? { backgroundColor: '#FFA500' } : {}}
                      />
                    </TableCell>
                    <TableCell>
                      {sample.true_label || 'Not validated'}
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                        {sample.raw_data_path && (
                          <Tooltip title="View Raw Data">
                            <IconButton size="small" sx={{ ml: 1 }}>
                              <ImageIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        {sample.feature_path && (
                          <Tooltip title="View Feature Vector">
                            <IconButton size="small" sx={{ ml: 1 }}>
                              <VectorIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Button
                        variant="outlined"
                        size="small"
                        disabled={sample.status === 'validated'}
                        onClick={() => handleOpenValidationDialog(sample)}
                      >
                        Validate
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Card>
      
      {/* Validation dialog */}
      <ValidationDialog
        open={validationDialogOpen}
        sample={selectedSample}
        onClose={handleCloseValidationDialog}
        onValidate={handleValidateSample}
      />
      
      {/* Status update dialog */}
      {event && (
        <StatusUpdateDialog
          open={statusDialogOpen}
          currentStatus={event.status}
          onClose={handleCloseStatusDialog}
          onUpdate={handleUpdateStatus}
        />
      )}
    </Box>
  );
};

export default DriftEventPage;