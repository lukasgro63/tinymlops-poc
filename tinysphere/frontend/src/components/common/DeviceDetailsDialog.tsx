import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Paper,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';
import React, { useEffect, useState } from 'react';
import { getPackagesByDevice } from '../../services/api';
import { Device, Package } from '../../types/api';

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

  useEffect(() => {
    if (open && device) {
      fetchDevicePackages(device.device_id);
    } else {
      setDevicePackages([]);
      setError(null);
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
                  <Typography variant="subtitle2" color="text.secondary">Platform</Typography>
                  <Typography variant="body1">{device.platform || 'Unknown'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">Python Version</Typography>
                  <Typography variant="body1">{device.python_version || 'N/A'}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 200px' }}>
                  <Typography variant="subtitle2" color="text.secondary">TinyLCM Version</Typography>
                  <Typography variant="body1">{device.tinylcm_version || 'N/A'}</Typography>
                </Box>
              </Box>
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
    </Dialog>
  );
};

export default DeviceDetailsDialog;