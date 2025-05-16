// src/pages/Devices.tsx
import DeviceIcon from '@mui/icons-material/Devices';
import DownloadIcon from '@mui/icons-material/Download';
import InfoIcon from '@mui/icons-material/Info';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import MemoryIcon from '@mui/icons-material/Memory';
import RefreshIcon from '@mui/icons-material/Refresh';
import SearchIcon from '@mui/icons-material/Search';
import SpeedIcon from '@mui/icons-material/Speed';
import StorageIcon from '@mui/icons-material/Storage';
import {
  Alert,
  Box,
  Button,
  Chip,
  FormControl,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
  Skeleton,
  Snackbar,
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
  Typography
} from '@mui/material';
import ErrorDisplay from '../components/common/ErrorDisplay';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ConnectivityTrendChart from '../components/common/ConnectivityTrendChart';
import DeviceDetailsDialog from '../components/common/DeviceDetailsDialog';
import DevicePerformanceChart from '../components/common/DevicePerformanceChart';
import DeviceStatusChart from '../components/common/DeviceStatusChart';
import PlatformDistributionChart from '../components/common/PlatformDistributionChart';
import SectionCard from '../components/common/SectionCard';
import StatusCard from '../components/common/StatusCard';
import TopDevicesChart from '../components/common/TopDevicesChart';
import {
  getDeviceConnectivityTrends,
  getDeviceMetrics,
  getDevicePlatforms,
  getDevices,
  getDevicesSummary,
  getTopDevices
} from '../services/api';
import {
  Device,
  DeviceMetrics,
  DeviceSummary,
  DeviceTrend,
  PlatformDistribution,
  TopDevice
} from '../types/api';

// Type für Sortierreihenfolge
type Order = 'asc' | 'desc';

// Type für sortierbare Spalten
type SortableKeys = 'hostname' | 'platform' | 'is_active' | 'last_sync_time';

const DevicesPage: React.FC = () => {
  // State für Device-Daten
  const [devices, setDevices] = useState<Device[]>([]);
  const [deviceSummaries, setDeviceSummaries] = useState<DeviceSummary[]>([]);
  const [platformData, setPlatformData] = useState<PlatformDistribution[]>([]);
  const [deviceMetrics, setDeviceMetrics] = useState<DeviceMetrics | null>(null);
  const [connectivityTrends, setConnectivityTrends] = useState<DeviceTrend[]>([]);
  const [topDevices, setTopDevices] = useState<TopDevice[]>([]);
  
  // UI State
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [platformFilter, setPlatformFilter] = useState<string>('all');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'info'>('info');
  
  // Sortierung und Paginierung
  const [order, setOrder] = useState<Order>('desc');
  const [orderBy, setOrderBy] = useState<SortableKeys>('last_sync_time');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  // Dialog State
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [openDialog, setOpenDialog] = useState<boolean>(false);
  
  // Automatische Aktualisierung
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Daten laden beim ersten Render
  useEffect(() => {
    fetchDevicesData();
  }, []);

  // Auto-Refresh einrichten
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (autoRefresh) {
      // Alle 30 Sekunden aktualisieren
      intervalId = setInterval(() => {
        fetchDevicesData();
      }, 30000);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh]);

  // Alle Device-Daten laden
  const fetchDevicesData = useCallback(async () => {
    setLoading(prev => !prev ? true : prev); // Nur setzen, wenn es vorher false war
    setError(null);
    
    try {
      // Alle Daten parallel laden
      const [
        devicesData,
        summariesData,
        platformsData,
        metricsData,
        trendsData,
        topDevicesData
      ] = await Promise.all([
        getDevices(),
        getDevicesSummary(),
        getDevicePlatforms(),
        getDeviceMetrics(),
        getDeviceConnectivityTrends(7),
        getTopDevices(5)
      ]);
      
      // State mit echten Daten aktualisieren
      setDevices(devicesData);
      setDeviceSummaries(summariesData);
      setPlatformData(platformsData);
      setDeviceMetrics(metricsData);
      setConnectivityTrends(trendsData);
      setTopDevices(topDevicesData);
      
      // Letzten Update-Zeitpunkt speichern
      setLastUpdated(new Date());
      
      // Erfolgsmeldung, wenn es kein initiales Laden war
      if (!loading) {
        showSnackbar('Data refreshed successfully', 'success');
      }
    } catch (error: any) {
      console.error('Error fetching devices data:', error);
      
      // Erweiterte Fehlerbehandlung
      if (error.response) {
        // Server-Antwort mit Fehlercode
        setError(`Server error: ${error.response.status} - ${error.response.data?.detail || 'Unknown error'}`);
      } else if (error.request) {
        // Keine Antwort vom Server
        setError('Network error: Unable to reach server. Please check your connection.');
      } else {
        setError(`Error: ${error.message || 'Unknown error'}`);
      }
      
      // Fehlermeldung anzeigen
      showSnackbar('Failed to load device data', 'error');
    } finally {
      setLoading(false);
    }
  }, [loading]);

  // Snackbar-Nachricht anzeigen
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  // Snackbar schließen
  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  // Sortieranforderung verarbeiten
  const handleRequestSort = (property: SortableKeys) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  // Geräte nach aktuellen Filtern und Sortierung filtern
  const filteredDevices = useMemo(() => {
    if (!devices.length) return [];
    
    let filtered = [...devices];
    
    // Suchfilter anwenden
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter(device => 
        device.device_id.toLowerCase().includes(searchLower) ||
        (device.hostname && device.hostname.toLowerCase().includes(searchLower)) ||
        (device.platform && device.platform.toLowerCase().includes(searchLower))
      );
    }
    
    // Statusfilter anwenden with 30-minute threshold
    if (statusFilter !== 'all') {
      const isActiveStatus = statusFilter === 'active';
      // Use the device's is_active property instead of calculating it again
      filtered = filtered.filter(device => {
        // For stability, just use the server-provided active status
        return device.is_active === isActiveStatus;
      });
    }
    
    // Plattformfilter anwenden
    if (platformFilter !== 'all') {
      filtered = filtered.filter(device => 
        device.platform && device.platform.toLowerCase() === platformFilter.toLowerCase()
      );
    }
    
    // Sortierung anwenden
    if (orderBy) {
      filtered.sort((a, b) => {
        let aValue: any;
        let bValue: any;

        // Use the default is_active property from server
        aValue = a[orderBy];
        bValue = b[orderBy];

        // Besondere Behandlung für verschiedene Typen
        if (orderBy === 'last_sync_time') {
          aValue = aValue ? new Date(aValue).getTime() : 0;
          bValue = bValue ? new Date(bValue).getTime() : 0;
        } else if (typeof aValue === 'string' && typeof bValue === 'string') {
          aValue = aValue.toLowerCase();
          bValue = bValue.toLowerCase();
        }

        if (aValue === bValue) return 0;

        // Sortierreihenfolge berücksichtigen
        const comparison = aValue < bValue ? -1 : 1;
        return order === 'asc' ? comparison : -comparison;
      });
    }
    
    return filtered;
  }, [devices, searchTerm, statusFilter, platformFilter, orderBy, order]);

  // CSV-Export-Funktion
  const exportDevicesData = () => {
    try {
      // CSV-Header und Daten erstellen
      const headers = ['Device ID', 'Hostname', 'Platform', 'Status', 'Last Sync', 'Packages'];
      const csvContent = [
        headers.join(','),
        ...filteredDevices.map(device => {
          const summary = deviceSummaries.find(s => s.device_id === device.device_id);
          return [
            device.device_id,
            device.hostname || 'N/A',
            device.platform || 'Unknown',
            device.is_active ? 'Active' : 'Inactive',
            device.last_sync_time ? new Date(device.last_sync_time).toLocaleString() : 'Never',
            summary?.package_count || 0
          ].join(',');
        })
      ].join('\n');
      
      // Download-Link erstellen
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `devices_export_${new Date().toISOString().slice(0, 10)}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      showSnackbar('Export completed successfully', 'success');
    } catch (error) {
      console.error('Export error:', error);
      showSnackbar('Failed to export device data', 'error');
    }
  };

  // Eindeutige Plattformen für Filter
  const uniquePlatforms = useMemo(() => {
    if (!devices || devices.length === 0) return [];
    
    const platforms = new Set<string>();
    devices.forEach(device => {
      if (device.platform) {
        platforms.add(device.platform);
      }
    });
    
    return Array.from(platforms);
  }, [devices]);

  // Gerätezählung für Status-Chart - using server's is_active property
  const activeCount = useMemo(() => devices.filter(d => d.is_active).length, [devices]);
  const inactiveCount = useMemo(() => devices.length - activeCount, [devices, activeCount]);

  // Diese Funktionen wurden entfernt, da sie für die neuen DevicePerformanceChart nicht mehr benötigt werden

  // Handler für Dialog
  const handleOpenDialog = (device: Device) => {
    setSelectedDevice(device);
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedDevice(null);
  };

  // Relative Zeit formatieren
  const formatRelativeTime = (dateString: string | undefined | null) => {
    if (!dateString) return 'Never';

    try {
      // Explicitly handle the Z suffix to ensure UTC parsing
      const dateStr = dateString.endsWith('Z') ? dateString : dateString + 'Z';

      // Parse the date string as UTC, then convert to local time
      const date = new Date(dateStr);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();

      const diffSeconds = Math.floor(diffMs / 1000);
      const diffMinutes = Math.floor(diffSeconds / 60);
      const diffHours = Math.floor(diffMinutes / 60);
      const diffDays = Math.floor(diffHours / 24);

      if (diffDays > 0) {
        return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
      } else if (diffHours > 0) {
        return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
      } else if (diffMinutes > 0) {
        return `${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago`;
      } else {
        return 'just now';
      }
    } catch (e) {
      console.error("Date parsing error:", e, "for date:", dateString);
      return 'Unknown';
    }
  };

  // Helper to check if device is active (synced within last 30 minutes)
  // Memoized to prevent recalculations on every render
  const isDeviceActive = useCallback((device: Device): boolean => {
    if (!device || !device.last_sync_time) return false;

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
  }, []);

  // Paginierung
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Loading or error state
  if (loading || error) {
    return (
      <ErrorDisplay 
        error={error}
        loading={loading}
        onRetry={() => fetchDevicesData()}
        height="70vh"
      />
    );
  }

  return (
    <Box>
      {/* Erste Reihe: Gerätestatistiken */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
        {/* Plattformverteilung */}
        <Box sx={{ flex: '1 1 350px', minWidth: '300px' }}>
          <SectionCard
            title="Platform Distribution"
            icon={<DeviceIcon style={{ fontSize: 20, color: '#0B2A5A' }} />}
            height={300}
          >
            <PlatformDistributionChart data={platformData} />
          </SectionCard>
        </Box>

        {/* Aktivitätsstatus */}
        <Box sx={{ flex: '1 1 350px', minWidth: '300px' }}>
          <SectionCard
            title="Activity Status"
            icon={<DeviceIcon style={{ fontSize: 20, color: '#0B2A5A' }} />}
            height={300}
          >
            <DeviceStatusChart activeCount={activeCount} inactiveCount={inactiveCount} />
          </SectionCard>
        </Box>

        {/* Konnektivitätstrend */}
        <Box sx={{ flex: '1 1 350px', minWidth: '300px' }}>
          <SectionCard
            title="Connectivity Trend (Last 7 Days)"
            icon={<DeviceIcon style={{ fontSize: 20, color: '#0B2A5A' }} />}
            height={300}
          >
            <ConnectivityTrendChart 
              data={
                // Wenn wir für heute Daten haben (meistens der Fall), ersetzen wir die aktive Anzahl
                // mit der aktuellen aktiven Geräteanzahl für eine konsistentere Darstellung
                connectivityTrends
                .map(day => {
                  const today = new Date().toISOString().split('T')[0];
                  
                  // Wenn es sich um den heutigen Tag handelt, verwenden wir die aktuelle Anzahl aktiver Geräte
                  // anstelle der vom Backend berechneten
                  if (day.date === today) {
                    return {
                      ...day,
                      active: activeCount,
                      inactive: day.total - activeCount
                    };
                  }
                  
                  // Für andere Tage: normale Berechnung
                  return {
                    ...day,
                    inactive: day.total - day.active,
                  };
                })
                // In umgekehrter Reihenfolge (neueste zuerst) sortieren
                .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                // Dann wieder umdrehen, damit der Graph von links nach rechts zeitlich korrekt ist
                .reverse()
              }
            />
          </SectionCard>
        </Box>
      </Box>

      {/* Zweite Reihe: Gerätemetriken-Dashboard und Top-Geräte */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
        {/* Geräte-Performance-Chart */}
        <Box sx={{ flex: '2 1 600px', minWidth: '600px' }}>
          <SectionCard
            title="Device Performance Metrics"
            icon={<SpeedIcon style={{ fontSize: 20, color: '#0B2A5A' }} />}
            height={400}
          >
            <DevicePerformanceChart />
          </SectionCard>
        </Box>

        {/* Top-Geräte */}
        <Box sx={{ flex: '1 1 350px', minWidth: '300px' }}>
          <SectionCard
            title="Top Devices by Package Count"
            icon={<LocalShippingIcon style={{ fontSize: 20, color: '#0B2A5A' }} />}
            height={400}
          >
            <TopDevicesChart data={topDevices} />
          </SectionCard>
        </Box>
      </Box>

      {/* Dritte Reihe: Gerätetabelle mit Filtern */}
      <Box sx={{ mb: 3 }}>
        <SectionCard
          title="Device List"
          icon={<DeviceIcon style={{ fontSize: 20, color: '#0B2A5A' }} />}
          action={
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Last updated">
                <Typography variant="caption" sx={{ alignSelf: 'center', mr: 1, color: 'text.secondary' }}>
                  {lastUpdated ? `Updated: ${lastUpdated.toLocaleTimeString()}` : ''}
                </Typography>
              </Tooltip>
              <Tooltip title="Refresh data">
                <IconButton size="small" onClick={() => fetchDevicesData()} disabled={loading}>
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Export to CSV">
                <IconButton 
                  size="small" 
                  onClick={exportDevicesData} 
                  disabled={filteredDevices.length === 0 || loading}
                >
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
            </Box>
          }
        >
          <Box sx={{ p: 2 }}>
            {/* Filterleiste */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
              <TextField
                placeholder="Search devices..."
                variant="outlined"
                size="small"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                sx={{ flex: '1 1 250px' }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
              />

              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel id="status-select-label">Status</InputLabel>
                <Select
                  labelId="status-select-label"
                  value={statusFilter}
                  label="Status"
                  onChange={(e: SelectChangeEvent) => setStatusFilter(e.target.value)}
                >
                  <MenuItem value="all">All Status</MenuItem>
                  <MenuItem value="active">Active</MenuItem>
                  <MenuItem value="inactive">Inactive</MenuItem>
                </Select>
              </FormControl>

              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel id="platform-select-label">Platform</InputLabel>
                <Select
                  labelId="platform-select-label"
                  value={platformFilter}
                  label="Platform"
                  onChange={(e: SelectChangeEvent) => setPlatformFilter(e.target.value)}
                >
                  <MenuItem value="all">All Platforms</MenuItem>
                  {uniquePlatforms.map(platform => (
                    <MenuItem key={platform} value={platform}>{platform}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* Gerätetabelle */}
            <TableContainer>
              <Table sx={{ minWidth: 650 }} size="medium">
                <TableHead>
                  <TableRow>
                    <TableCell>
                      <TableSortLabel
                        active={orderBy === 'hostname'}
                        direction={orderBy === 'hostname' ? order : 'asc'}
                        onClick={() => handleRequestSort('hostname')}
                      >
                        Device
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>
                      <TableSortLabel
                        active={orderBy === 'platform'}
                        direction={orderBy === 'platform' ? order : 'asc'}
                        onClick={() => handleRequestSort('platform')}
                      >
                        Platform
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>
                      <TableSortLabel
                        active={orderBy === 'is_active'}
                        direction={orderBy === 'is_active' ? order : 'asc'}
                        onClick={() => handleRequestSort('is_active')}
                      >
                        Status
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>
                      <TableSortLabel
                        active={orderBy === 'last_sync_time'}
                        direction={orderBy === 'last_sync_time' ? order : 'asc'}
                        onClick={() => handleRequestSort('last_sync_time')}
                      >
                        Last Sync
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>Packages</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredDevices.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography variant="body1" sx={{ py: 2 }}>No devices found matching your filters</Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredDevices
                      .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                      .map((device) => {
                        // Zugehörige DeviceSummary für zusätzliche Informationen finden
                        const summary = deviceSummaries.find(d => d.device_id === device.device_id);
                        
                        return (
                          <TableRow key={device.device_id} hover>
                            <TableCell>
                              <Box>
                                <Typography variant="body2" fontWeight="medium">{device.hostname || 'N/A'}</Typography>
                                <Typography variant="caption" color="text.secondary">{device.device_id}</Typography>
                              </Box>
                            </TableCell>
                            <TableCell>{device.platform || 'Unknown'}</TableCell>
                            <TableCell>
                              <Chip
                                label={device.is_active ? 'Active' : 'Inactive'}
                                size="small"
                                color={device.is_active ? 'success' : 'warning'}
                                sx={device.is_active ? {} : { backgroundColor: '#FFA500' }}
                              />
                            </TableCell>
                            <TableCell>{formatRelativeTime(device.last_sync_time)}</TableCell>
                            <TableCell>{summary?.package_count || 0}</TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', gap: 1 }}>
                                <Tooltip title="View Details">
                                  <IconButton 
                                    size="small" 
                                    onClick={() => handleOpenDialog(device)}
                                    data-test-id="info-button"
                                  >
                                    <InfoIcon fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              </Box>
                            </TableCell>
                          </TableRow>
                        );
                      })
                  )}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Paginierung */}
            <TablePagination
              rowsPerPageOptions={[10, 25, 50, 100]}
              component="div"
              count={filteredDevices.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </Box>
        </SectionCard>
      </Box>

      {/* Detaildialog */}
      <DeviceDetailsDialog 
        device={selectedDevice}
        open={openDialog}
        onClose={handleCloseDialog}
      />
      
      {/* Snackbar für Benachrichtigungen */}
      <Snackbar 
        open={snackbarOpen} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbarSeverity}
          variant="filled"
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DevicesPage;