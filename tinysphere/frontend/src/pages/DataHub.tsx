import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Card, 
  CardMedia, 
  CardContent, 
  IconButton, 
  Dialog, 
  DialogContent, 
  DialogTitle,
  Tab,
  Tabs,
  FormControl, 
  InputLabel, 
  Select, 
  SelectChangeEvent,
  MenuItem,
  TablePagination, 
  Chip, 
  Divider, 
  Tooltip, 
  CircularProgress, 
  FormHelperText,
  Button
} from '@mui/material';
import { 
  Storage as DataHubIcon,
  CameraAlt as CameraIcon,
  Refresh as RefreshIcon, 
  FilterList as FilterIcon, 
  Close as CloseIcon, 
  DownloadForOffline as DownloadIcon,
  SmartToy as DeviceIcon,
  Image as GalleryIcon,
  List as ListIcon,
  Dashboard as DashboardIcon
} from '@mui/icons-material';
import { format, parseISO } from 'date-fns';

import { getDevices, getPredictionImageDevices, getPredictionTypes, getPredictionDates, getPredictionImages } from '../services/api';
import { Device, PredictionImage, PredictionImagesResponse } from '../types/api';
import ErrorDisplay from '../components/common/ErrorDisplay';

// Tab panel component for the content sections
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`data-hub-tabpanel-${index}`}
      aria-labelledby={`data-hub-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const DataHub: React.FC = () => {
  // State for tab control
  const [tabValue, setTabValue] = useState(0);
  
  // State for device selection
  const [devices, setDevices] = useState<Device[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('all');
  const [loadingDevices, setLoadingDevices] = useState(true);
  
  // State for prediction images
  const [images, setImages] = useState<PredictionImage[]>([]);
  const [totalImages, setTotalImages] = useState(0);
  const [loadingImages, setLoadingImages] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // State for image filters
  const [predictionTypes, setPredictionTypes] = useState<string[]>([]);
  const [dates, setDates] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>('');
  
  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(16);
  
  // Selected image for details view
  const [selectedImage, setSelectedImage] = useState<PredictionImage | null>(null);
  
  // Load devices on initial render
  useEffect(() => {
    const loadDevices = async () => {
      setLoadingDevices(true);
      try {
        const allDevices = await getDevices();
        setDevices(allDevices);
        
        // If devices are available, get available device IDs with prediction images
        const deviceIdsWithImages = await getPredictionImageDevices();
        
        // Filter devices to only those with prediction images
        const devicesWithImages = allDevices.filter(device => 
          deviceIdsWithImages.includes(device.device_id)
        );
        
        // If no devices with images, clear selection
        if (devicesWithImages.length === 0) {
          setSelectedDeviceId('');
        }
      } catch (err) {
        console.error('Error loading devices:', err);
        setError('Failed to load devices');
      } finally {
        setLoadingDevices(false);
      }
    };
    
    loadDevices();
  }, []);
  
  // Load prediction types when device is selected
  useEffect(() => {
    const loadPredictionTypes = async () => {
      if (!selectedDeviceId || selectedDeviceId === 'all') {
        setPredictionTypes([]);
        setSelectedType('');
        return;
      }
      
      try {
        const types = await getPredictionTypes(selectedDeviceId);
        setPredictionTypes(types);
        
        // Auto-select first type if available
        if (types.length > 0) {
          setSelectedType(types[0]);
        } else {
          setSelectedType('');
        }
      } catch (err) {
        console.error('Error loading prediction types:', err);
        setError('Failed to load prediction types');
      }
    };
    
    loadPredictionTypes();
  }, [selectedDeviceId]);
  
  // Load dates when prediction type is selected
  useEffect(() => {
    const loadDates = async () => {
      if (!selectedDeviceId || selectedDeviceId === 'all' || !selectedType) {
        setDates([]);
        setSelectedDate('');
        return;
      }
      
      try {
        const datesList = await getPredictionDates(selectedDeviceId, selectedType);
        setDates(datesList);
        
        // Auto-select first date if available
        if (datesList.length > 0) {
          setSelectedDate(datesList[0]);
        } else {
          setSelectedDate('');
        }
      } catch (err) {
        console.error('Error loading dates:', err);
        setError('Failed to load dates');
      }
    };
    
    loadDates();
  }, [selectedDeviceId, selectedType]);
  
  // Load images based on filters and pagination
  useEffect(() => {
    const loadImages = async () => {
      // Only load images on the Gallery tab
      if (tabValue !== 1) return;
      
      setLoadingImages(true);
      setError(null);
      
      try {
        if (selectedDeviceId && selectedDeviceId !== 'all') {
          // For a specific device
          const response = await getPredictionImages(
            selectedDeviceId,
            selectedType || undefined,
            selectedDate || undefined,
            rowsPerPage,
            page * rowsPerPage
          );
          
          setImages(response.images);
          setTotalImages(response.total);
        } else {
          // For "All Devices" option - we need to fetch for each device with images
          const deviceIdsWithImages = await getPredictionImageDevices();
          let allImages: PredictionImage[] = [];
          
          // This is a simplified approach - in a real implementation, you might
          // want to add pagination and more sophisticated fetching
          for (const deviceId of deviceIdsWithImages) {
            const response = await getPredictionImages(
              deviceId,
              selectedType || undefined,
              selectedDate || undefined,
              rowsPerPage,
              0 // For simplicity, just get the first page from each device
            );
            
            allImages = [...allImages, ...response.images];
          }
          
          // Simple client-side pagination
          const startIndex = page * rowsPerPage;
          const endIndex = startIndex + rowsPerPage;
          
          setTotalImages(allImages.length);
          setImages(allImages.slice(startIndex, endIndex));
        }
      } catch (err) {
        console.error('Error loading images:', err);
        setError('Failed to load images');
        setImages([]);
        setTotalImages(0);
      } finally {
        setLoadingImages(false);
      }
    };
    
    loadImages();
  }, [selectedDeviceId, selectedType, selectedDate, page, rowsPerPage, tabValue]);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Handle device selection
  const handleDeviceChange = (event: SelectChangeEvent<string>) => {
    const deviceId = event.target.value;
    setSelectedDeviceId(deviceId);
    setSelectedType('');
    setSelectedDate('');
    setPage(0);
  };
  
  // Handle prediction type selection
  const handleTypeChange = (event: SelectChangeEvent<string>) => {
    const type = event.target.value;
    setSelectedType(type);
    setSelectedDate('');
    setPage(0);
  };
  
  // Handle date selection
  const handleDateChange = (event: SelectChangeEvent<string>) => {
    const date = event.target.value;
    setSelectedDate(date);
    setPage(0);
  };
  
  // Handle pagination
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };
  
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  // Handle image selection for details view
  const handleImageClick = (image: PredictionImage) => {
    setSelectedImage(image);
  };
  
  // Close details view
  const handleCloseDetails = () => {
    setSelectedImage(null);
  };
  
  // Download image
  const handleDownload = (imageKey: string, filename: string) => {
    // Create a download URL by adding the download parameter
    const downloadUrl = `/api/prediction-images/image/${imageKey}?download=true`;

    // Create a link and click it to start the download
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  // Format a date string from timestamp format (20220325) to readable format (Mar 25, 2022)
  const formatDateString = (dateStr: string) => {
    if (!dateStr || dateStr.length !== 8) return dateStr;
    
    try {
      const year = dateStr.substring(0, 4);
      const month = dateStr.substring(4, 6);
      const day = dateStr.substring(6, 8);
      
      return format(new Date(`${year}-${month}-${day}`), 'MMM d, yyyy');
    } catch (err) {
      return dateStr;
    }
  };
  
  // Render device selection
  const renderDeviceSelector = () => (
    <Box sx={{ mb: 3 }}>
      <FormControl fullWidth size="small">
        <InputLabel id="device-select-label">Device</InputLabel>
        <Select
          labelId="device-select-label"
          value={selectedDeviceId}
          label="Device"
          onChange={handleDeviceChange}
        >
          <MenuItem value="all">All Devices</MenuItem>
          {devices.map(device => (
            <MenuItem key={device.device_id} value={device.device_id}>
              {device.device_id} {device.hostname ? `(${device.hostname})` : ''}
            </MenuItem>
          ))}
        </Select>
        <FormHelperText>Select a device to view its data</FormHelperText>
      </FormControl>
    </Box>
  );
  
  // Render prediction images gallery
  const renderImageGallery = () => {
    if (loadingImages) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (error) {
      return (
        <ErrorDisplay 
          error={error}
          loading={false}
          onRetry={() => setLoadingImages(true)}
          height="50vh"
        />
      );
    }
    
    if (images.length === 0) {
      return (
        <Box sx={{ textAlign: 'center', p: 4 }}>
          <Typography variant="body1" color="text.secondary">
            No prediction images found for the selected filters.
          </Typography>
          {!selectedDeviceId && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Please select a device to view available images.
            </Typography>
          )}
        </Box>
      );
    }
    
    return (
      <Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {images.map((image) => (
            <Box 
              key={image.key}
              sx={{ 
                flex: '1 1 calc(25% - 16px)',
                minWidth: '250px',
                maxWidth: 'calc(25% - 16px)'
              }}
            >
              <Card 
                sx={{ 
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  '&:hover': { 
                    transform: 'scale(1.02)',
                    boxShadow: 3
                  }
                }}
                onClick={() => handleImageClick(image)}
              >
                <CardMedia
                  component="img"
                  sx={{ 
                    height: 180,
                    objectFit: 'cover',
                    bgcolor: 'rgba(0,0,0,0.05)'
                  }}
                  image={image.url}
                  alt={image.filename}
                />
                <CardContent sx={{ pb: 1, pt: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" noWrap title={image.filename}>
                      {image.filename}
                    </Typography>
                    <Chip 
                      label={image.prediction_type} 
                      size="small" 
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {formatDateString(image.date)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {image.device_id}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>
        
        {/* Pagination */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <TablePagination
            component="div"
            count={totalImages}
            page={page}
            onPageChange={handleChangePage}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={handleChangeRowsPerPage}
            rowsPerPageOptions={[12, 16, 24, 48]}
          />
        </Box>
      </Box>
    );
  };
  
  // Render image filters
  const renderImageFilters = () => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <FormControl fullWidth size="small" disabled={!selectedDeviceId || predictionTypes.length === 0}>
            <InputLabel id="type-select-label">Prediction Type</InputLabel>
            <Select
              labelId="type-select-label"
              value={selectedType}
              label="Prediction Type"
              onChange={handleTypeChange}
            >
              {predictionTypes.map(type => (
                <MenuItem key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <FormControl fullWidth size="small" disabled={!selectedType || dates.length === 0}>
            <InputLabel id="date-select-label">Date</InputLabel>
            <Select
              labelId="date-select-label"
              value={selectedDate}
              label="Date"
              onChange={handleDateChange}
            >
              {dates.map(date => (
                <MenuItem key={date} value={date}>{formatDateString(date)}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Box sx={{ flex: '1 1 200px', display: 'flex', alignItems: 'center' }}>
          <Button 
            variant="outlined" 
            disabled={!selectedType && !selectedDate}
            onClick={() => {
              setSelectedType('');
              setSelectedDate('');
            }}
          >
            Clear Filters
          </Button>
        </Box>
      </Box>
    </Box>
  );
  
  // Render a placeholder for other data sections
  const renderDataPlaceholder = (title: string) => (
    <Box sx={{ textAlign: 'center', p: 4, bgcolor: 'background.paper', borderRadius: 1 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <Typography variant="body1" color="text.secondary">
        This section is under development.
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        Select a device from the dropdown above to view its data.
      </Typography>
    </Box>
  );
  
  return (
    <Box sx={{ p: 0 }}>
      {/* Header */}
      <Paper sx={{ mb: 3, p: 2 }} elevation={2}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <DataHubIcon style={{ fontSize: 24, color: '#00647D', marginRight: '10px' }} />
            <Typography variant="h5" component="h1">
              Data Hub
            </Typography>
          </Box>
          
          <Box>
            <Tooltip title="Refresh">
              <IconButton onClick={() => {
                // Refresh current view
                setPage(0);
                setLoadingImages(true);
              }}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        {/* Device selector */}
        {renderDeviceSelector()}
        
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="data hub tabs"
            indicatorColor="primary"
            textColor="primary"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab 
              icon={<DashboardIcon />} 
              iconPosition="start" 
              label="Overview" 
            />
            <Tab 
              icon={<GalleryIcon />} 
              iconPosition="start" 
              label="Prediction Images" 
              disabled={!selectedDeviceId}
            />
            <Tab 
              icon={<ListIcon />} 
              iconPosition="start" 
              label="Logs" 
              disabled={!selectedDeviceId}
            />
            <Tab 
              icon={<DeviceIcon />} 
              iconPosition="start" 
              label="Device Data" 
              disabled={!selectedDeviceId}
            />
          </Tabs>
        </Box>
      </Paper>
      
      {/* Tab content */}
      <TabPanel value={tabValue} index={0}>
        {renderDataPlaceholder("Device Data Overview")}
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        {/* Filters for prediction images */}
        {selectedDeviceId && renderImageFilters()}
        
        {/* Images gallery */}
        {renderImageGallery()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        {renderDataPlaceholder("Device Logs")}
      </TabPanel>
      
      <TabPanel value={tabValue} index={3}>
        {renderDataPlaceholder("Device Telemetry Data")}
      </TabPanel>
      
      {/* Image detail dialog */}
      {selectedImage && (
        <Dialog
          open={true}
          onClose={handleCloseDetails}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6">{selectedImage.filename}</Typography>
              <Box>
                <Tooltip title="Download">
                  <IconButton onClick={() => handleDownload(selectedImage.key, selectedImage.filename)}>
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Close">
                  <IconButton onClick={handleCloseDetails} edge="end">
                    <CloseIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          </DialogTitle>
          <DialogContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                bgcolor: 'rgba(0,0,0,0.03)',
                p: 2,
                borderRadius: 1,
                mb: 2
              }}>
                <img 
                  src={selectedImage.url} 
                  alt={selectedImage.filename} 
                  style={{ 
                    maxWidth: '100%',
                    maxHeight: '60vh',
                    objectFit: 'contain'
                  }}
                />
              </Box>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mt: 1 }}>
                <Box sx={{ flex: '1 1 45%', minWidth: '200px' }}>
                  <Typography variant="subtitle2">Device ID</Typography>
                  <Typography variant="body2">{selectedImage.device_id}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 45%', minWidth: '200px' }}>
                  <Typography variant="subtitle2">Prediction Type</Typography>
                  <Typography variant="body2">{selectedImage.prediction_type}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 45%', minWidth: '200px' }}>
                  <Typography variant="subtitle2">Date</Typography>
                  <Typography variant="body2">{formatDateString(selectedImage.date)}</Typography>
                </Box>
                <Box sx={{ flex: '1 1 45%', minWidth: '200px' }}>
                  <Typography variant="subtitle2">File Size</Typography>
                  <Typography variant="body2">{Math.round(selectedImage.size / 1024)} KB</Typography>
                </Box>
                <Box sx={{ flex: '1 1 100%' }}>
                  <Typography variant="subtitle2">Last Modified</Typography>
                  <Typography variant="body2">
                    {format(parseISO(selectedImage.last_modified), 'MMM d, yyyy HH:mm:ss')}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </DialogContent>
        </Dialog>
      )}
    </Box>
  );
};

export default DataHub;