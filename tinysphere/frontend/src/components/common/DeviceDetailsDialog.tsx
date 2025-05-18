import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Paper,
  Skeleton,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
  Alert
} from '@mui/material';
import React, { useEffect, useState } from 'react';
import { getPackagesByDevice, updateDeviceLocation } from '../../services/api';
import { Device, Package } from '../../types/api';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';

// Helper to check if device is active (synced within last 30 minutes)
const isDeviceActive = (device: Device): boolean => {
  if (!device.last_sync_time) return false;

  try {
    // Explicitly handle the Z suffix to ensure UTC parsing
    const dateStr = device.last_sync_time.endsWith('Z') ? device.last_sync_time : device.last_sync_time + 'Z';

    // Parse the date string as UTC
    const syncTime = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - syncTime.getTime();

    // Convert to minutes and check against 30-minute threshold
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    return diffMinutes < 30;
  } catch (e) {
    console.error("Date parsing error:", e, "for date:", device.last_sync_time);
    return false;
  }
};

interface DeviceDetailsDialogProps {
  device: Device | null;
  open: boolean;
  onClose: () => void;
}

const DeviceDetailsDialog: React.FC<DeviceDetailsDialogProps> = ({ 
  device, 
  open, 
  onClose 
}) => {
  const [devicePackages, setDevicePackages] = useState<Package[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Location editing state
  const [editingLocation, setEditingLocation] = useState(false);
  const [latitude, setLatitude] = useState<string>('');
  const [longitude, setLongitude] = useState<string>('');
  const [locationName, setLocationName] = useState<string>('');
  const [locationError, setLocationError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  
  // Snackbar state
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'info'>('info');

  useEffect(() => {
    if (open && device) {
      fetchDevicePackages(device.device_id);
      
      // Initialize location form values when the dialog opens
      if (device.latitude) setLatitude(device.latitude.toString());
      if (device.longitude) setLongitude(device.longitude.toString());
      if (device.location_name) setLocationName(device.location_name);
    } else {
      setDevicePackages([]);
      setError(null);
      
      // Reset location editing state
      setEditingLocation(false);
      setLocationError(null);
    }
  }, [open, device]);

  const fetchDevicePackages = async (deviceId: string) => {
    setLoading(true);
    setError(null);
    try {
      const packages = await getPackagesByDevice(deviceId);
      setDevicePackages(packages);
    } catch (error: any) {
      console.error('Error fetching device packages:', error);
      setError(error.message || 'Failed to load packages data');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle location editing
  const startLocationEdit = () => {
    setEditingLocation(true);
    setLocationError(null);
  };
  
  const cancelLocationEdit = () => {
    // Reset form values
    if (device) {
      setLatitude(device.latitude ? device.latitude.toString() : '');
      setLongitude(device.longitude ? device.longitude.toString() : '');
      setLocationName(device.location_name || '');
    }
    setEditingLocation(false);
    setLocationError(null);
  };
  
  const saveLocation = async () => {
    if (!device) return;
    
    // Validation
    if (!latitude || !longitude) {
      setLocationError('Latitude and longitude are required');
      return;
    }
    
    const latNum = parseFloat(latitude);
    const lngNum = parseFloat(longitude);
    
    if (isNaN(latNum) || isNaN(lngNum)) {
      setLocationError('Latitude and longitude must be valid numbers');
      return;
    }
    
    if (latNum < -90 || latNum > 90) {
      setLocationError('Latitude must be between -90 and 90');
      return;
    }
    
    if (lngNum < -180 || lngNum > 180) {
      setLocationError('Longitude must be between -180 and 180');
      return;
    }
    
    setSaving(true);
    setLocationError(null);
    
    try {
      await updateDeviceLocation(device.device_id, {
        latitude: latNum,
        longitude: lngNum,
        location_name: locationName.trim() || undefined
      });
      
      // Update the device object locally to reflect changes
      device.latitude = latNum;
      device.longitude = lngNum;
      device.location_name = locationName.trim() || undefined;
      device.last_location_update = new Date().toISOString();
      
      setEditingLocation(false);
      showSnackbar('Location updated successfully', 'success');
    } catch (error: any) {
      console.error('Error updating device location:', error);
      setLocationError(error.message || 'Failed to update location');
      showSnackbar('Failed to update location', 'error');
    } finally {
      setSaving(false);
    }
  };
  
  // Show snackbar message
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };
  
  // Close snackbar
  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  if (!device) {
    return null;
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        Device Details: {device.hostname || device.device_id}
      </DialogTitle>
      <DialogContent dividers>
        {loading ? (
          <Box sx={{ p: 2 }}>
            <Skeleton variant="rectangular" height={80} sx={{ mb: 2 }} />
            <Skeleton variant="rectangular" height={200} />
          </Box>
        ) : error ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography color="error">{error}</Typography>
            <Button 
              variant="contained" 
              sx={{ mt: 2 }}
              onClick={() => fetchDevicePackages(device.device_id)}
            >
              Retry
            </Button>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Basic information */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Basic Information</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Device ID</Typography>
                  <Typography variant="body1">{device.device_id}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Hostname</Typography>
                  <Typography variant="body1">{device.hostname || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">IP Address</Typography>
                  <Typography variant="body1">{device.ip_address || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Status</Typography>
                  <Typography variant="body1">
                    <Chip
                      label={device.is_active ? 'Active' : 'Inactive'}
                      size="small"
                      color={device.is_active ? 'success' : 'warning'}
                      sx={device.is_active ? {} : { backgroundColor: '#FFA500' }}
                    />
                  </Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">OS</Typography>
                  <Typography variant="body1">{device.platform || 'Unknown'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">OS Version</Typography>
                  <Typography variant="body1">{device.platform_version || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Device Model</Typography>
                  <Typography variant="body1">{device.device_model || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Python Version</Typography>
                  <Typography variant="body1">{device.python_version || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">TinyLCM Version</Typography>
                  <Typography variant="body1">{device.tinylcm_version || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Last Sync Time</Typography>
                  <Typography variant="body1">
                    {device.last_sync_time
                      ? new Date(device.last_sync_time).toLocaleString()
                      : 'Never'
                    }
                  </Typography>
                </Box>
              </Box>
            </Paper>
            
            {/* Location information */}
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Location Information</Typography>
                {!editingLocation ? (
                  <Tooltip title="Edit Location">
                    <IconButton size="small" onClick={startLocationEdit}>
                      <EditIcon />
                    </IconButton>
                  </Tooltip>
                ) : (
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="Save">
                      <IconButton size="small" onClick={saveLocation} disabled={saving}>
                        <SaveIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Cancel">
                      <IconButton size="small" onClick={cancelLocationEdit} disabled={saving}>
                        <CancelIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                )}
              </Box>
              
              {editingLocation ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    <TextField
                      label="Latitude"
                      value={latitude}
                      onChange={(e) => setLatitude(e.target.value)}
                      sx={{ flex: '1 1 200px' }}
                      size="small"
                      disabled={saving}
                      error={locationError !== null}
                      helperText={locationError}
                      required
                    />
                    <TextField
                      label="Longitude"
                      value={longitude}
                      onChange={(e) => setLongitude(e.target.value)}
                      sx={{ flex: '1 1 200px' }}
                      size="small"
                      disabled={saving}
                      required
                    />
                    <TextField
                      label="Location Name"
                      value={locationName}
                      onChange={(e) => setLocationName(e.target.value)}
                      sx={{ flex: '1 1 400px' }}
                      size="small"
                      placeholder="e.g., Headquarters, Production Site, etc."
                      disabled={saving}
                    />
                  </Box>
                  {locationError && (
                    <Typography color="error" variant="caption">
                      {locationError}
                    </Typography>
                  )}
                </Box>
              ) : device.latitude && device.longitude ? (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                  <Box sx={{ flex: '1 1 200px' }}>
                    <Typography variant="subtitle2" color="text.secondary">Latitude</Typography>
                    <Typography variant="body1">{device.latitude.toFixed(6)}</Typography>
                  </Box>
                  <Box sx={{ flex: '1 1 200px' }}>
                    <Typography variant="subtitle2" color="text.secondary">Longitude</Typography>
                    <Typography variant="body1">{device.longitude.toFixed(6)}</Typography>
                  </Box>
                  <Box sx={{ flex: '1 1 200px' }}>
                    <Typography variant="subtitle2" color="text.secondary">Location Name</Typography>
                    <Typography variant="body1">{device.location_name || 'N/A'}</Typography>
                  </Box>
                  <Box sx={{ flex: '1 1 200px' }}>
                    <Typography variant="subtitle2" color="text.secondary">Last Location Update</Typography>
                    <Typography variant="body1">
                      {device.last_location_update
                        ? new Date(device.last_location_update).toLocaleString()
                        : 'Never'
                      }
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Box>
                  <Typography variant="body2" color="text.secondary" sx={{ py: 1 }}>
                    No location data available for this device. 
                    Click the edit button to add location information.
                  </Typography>
                </Box>
              )}
            </Paper>

            {/* Package list */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Uploaded Packages</Typography>
              {devicePackages.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
                  No packages found for this device
                </Typography>
              ) : (
                <TableContainer sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Package ID</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Uploaded</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {devicePackages.map(pkg => (
                        <TableRow key={pkg.package_id}>
                          <TableCell>{pkg.package_id}</TableCell>
                          <TableCell>{pkg.package_type}</TableCell>
                          <TableCell>{new Date(pkg.uploaded_at).toLocaleString()}</TableCell>
                          <TableCell>
                            <Chip 
                              label={pkg.processing_status} 
                              size="small"
                              color={
                                pkg.processing_status === 'success' ? 'success' : 
                                pkg.processing_status === 'error' ? 'error' : 'warning'
                              }
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>

            {/* Device information as JSON */}
            {device.device_info && (
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>Device Info</Typography>
                <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                  <pre style={{ margin: 0 }}>
                    {JSON.stringify(device.device_info, null, 2)}
                  </pre>
                </Box>
              </Paper>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={snackbarOpen} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbarSeverity}
          variant="filled"
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Dialog>
  );
};

export default DeviceDetailsDialog;