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
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField
} from '@mui/material';
import {
  Storage as DataHubIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Close as CloseIcon,
  DownloadForOffline as DownloadIcon,
  Image as GalleryIcon,
  List as ListIcon,
  WarningAmber,
  Sort as SortIcon,
  ArrowUpward as SortAscIcon,
  ArrowDownward as SortDescIcon,
  DataObject as LogFileIcon,
  DeleteOutline as DeleteIcon
} from '@mui/icons-material';
import { format, parseISO } from 'date-fns';

import {
  getDevices,
  getPredictionImageDevices,
  getPredictionTypes,
  getPredictionDates,
  getPredictionImages,
  getDriftImageDevices,
  getDriftTypes,
  getDriftDates,
  getDriftImages,
  getOperationalLogDevices,
  getOperationalLogTypes,
  getOperationalLogs,
  deleteDeviceLogs,
  deleteSessionLogs
} from '../services/api';
import {
  Device,
  PredictionImage,
  PredictionImagesResponse,
  DriftImage,
  DriftImagesResponse,
  OperationalLog,
  OperationalLogResponse
} from '../types/api';
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
  const [predictionStartDate, setPredictionStartDate] = useState<string>('');
  const [predictionEndDate, setPredictionEndDate] = useState<string>('');

  // State for drift images
  const [driftImages, setDriftImages] = useState<DriftImage[]>([]);
  const [totalDriftImages, setTotalDriftImages] = useState(0);
  const [loadingDriftImages, setLoadingDriftImages] = useState(false);
  const [driftError, setDriftError] = useState<string | null>(null);
  const [driftStartDate, setDriftStartDate] = useState<string>('');
  const [driftEndDate, setDriftEndDate] = useState<string>('');
  
  // State for operational logs
  const [operationalLogs, setOperationalLogs] = useState<OperationalLog[]>([]);
  const [totalLogs, setTotalLogs] = useState(0);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [logTypes, setLogTypes] = useState<string[]>([]);
  const [selectedLogType, setSelectedLogType] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  // State for image filters
  const [predictionTypes, setPredictionTypes] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState<string>('');

  // State for drift image filters
  const [driftTypes, setDriftTypes] = useState<string[]>([]);
  const [selectedDriftType, setSelectedDriftType] = useState<string>('');
  
  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(16);
  
  // Sorting
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc'); // desc = newest first (default)
  
  // Selected image for details view
  const [selectedImage, setSelectedImage] = useState<PredictionImage | null>(null);
  
  // State for delete confirmation dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteType, setDeleteType] = useState<'device' | 'session'>('device');
  const [logToDelete, setLogToDelete] = useState<{deviceId: string; sessionId?: string} | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  
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
      try {
        let types: string[] = [];
        
        if (!selectedDeviceId || selectedDeviceId === 'all') {
          // For "All Devices", collect prediction types from all devices with images
          const deviceIdsWithImages = await getPredictionImageDevices();
          const allTypes = new Set<string>();
          
          // Get types from each device and combine them
          for (const deviceId of deviceIdsWithImages) {
            try {
              const deviceTypes = await getPredictionTypes(deviceId);
              deviceTypes.forEach(type => allTypes.add(type));
            } catch (e) {
              console.warn(`Failed to load prediction types for device ${deviceId}:`, e);
            }
          }
          
          types = Array.from(allTypes);
          console.log(`Loaded ${types.length} combined prediction types for all devices`);
        } else {
          // For a specific device
          types = await getPredictionTypes(selectedDeviceId);
          console.log(`Loaded ${types.length} prediction types for device ${selectedDeviceId}`);
        }
        
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

  // Load drift types when device is selected
  useEffect(() => {
    const loadDriftTypes = async () => {
      try {
        let types: string[] = [];
        
        if (!selectedDeviceId || selectedDeviceId === 'all') {
          // For "All Devices", collect drift types from all devices with drift images
          const deviceIdsWithDriftImages = await getDriftImageDevices();
          const allTypes = new Set<string>();
          
          // Get types from each device and combine them
          for (const deviceId of deviceIdsWithDriftImages) {
            try {
              const deviceTypes = await getDriftTypes(deviceId);
              deviceTypes.forEach(type => allTypes.add(type));
            } catch (e) {
              console.warn(`Failed to load drift types for device ${deviceId}:`, e);
            }
          }
          
          types = Array.from(allTypes);
          console.log(`Loaded ${types.length} combined drift types for all devices`);
        } else {
          // For a specific device
          types = await getDriftTypes(selectedDeviceId);
          console.log(`Loaded ${types.length} drift types for device ${selectedDeviceId}`);
        }
        
        setDriftTypes(types);
        
        // Auto-select first type if available
        if (types.length > 0) {
          setSelectedDriftType(types[0]);
        } else {
          setSelectedDriftType('');
        }
      } catch (err) {
        console.error('Error loading drift types:', err);
        setDriftError('Failed to load drift types');
      }
    };
    
    // Only load drift types when the drift tab is active
    if (tabValue === 1) {
      loadDriftTypes();
    }
  }, [selectedDeviceId, tabValue]);
  
  // Load operational log types when device is selected
  useEffect(() => {
    const loadLogTypes = async () => {
      try {
        let types: string[] = [];
        
        if (!selectedDeviceId || selectedDeviceId === 'all') {
          // For "All Devices", collect log types from all devices with logs
          const deviceIdsWithLogs = await getOperationalLogDevices();
          const allTypes = new Set<string>();
          
          // Get types from each device and combine them
          for (const deviceId of deviceIdsWithLogs) {
            try {
              const deviceTypes = await getOperationalLogTypes(deviceId);
              deviceTypes.forEach(type => allTypes.add(type));
            } catch (e) {
              console.warn(`Failed to load log types for device ${deviceId}:`, e);
            }
          }
          
          types = Array.from(allTypes);
          console.log(`Loaded ${types.length} combined log types for all devices`);
        } else {
          // For a specific device
          types = await getOperationalLogTypes(selectedDeviceId);
          console.log(`Loaded ${types.length} log types for device ${selectedDeviceId}`);
        }
        
        setLogTypes(types);
        
        // Auto-select first type if available
        if (types.length > 0) {
          setSelectedLogType(types[0]);
        } else {
          setSelectedLogType('');
        }
      } catch (err) {
        console.error('Error loading log types:', err);
        setLogsError('Failed to load log types');
      }
    };
    
    // Only load log types when the logs tab is active
    if (tabValue === 2) {
      loadLogTypes();
    }
  }, [selectedDeviceId, tabValue]);
  
  // No longer need to load dates for dropdown
  useEffect(() => {
    // Reset any date filters when device or prediction type changes
    if (!selectedDeviceId || selectedDeviceId === 'all' || !selectedType) {
      setPredictionStartDate('');
      setPredictionEndDate('');
    }
  }, [selectedDeviceId, selectedType]);

  // No longer need to load dates for dropdown
  useEffect(() => {
    // Reset any date filters when device or drift type changes
    if (!selectedDeviceId || selectedDeviceId === 'all' || !selectedDriftType) {
      setDriftStartDate('');
      setDriftEndDate('');
    }
  }, [selectedDeviceId, selectedDriftType, tabValue]);
  
  // Load images based on filters and pagination
  useEffect(() => {
    const loadImages = async () => {
      // Only load images on the Gallery tab
      if (tabValue !== 0) return;

      setLoadingImages(true);
      setError(null);
      
      try {
        if (selectedDeviceId && selectedDeviceId !== 'all') {
          // For a specific device
          const response = await getPredictionImages(
            selectedDeviceId,
            selectedType || undefined,
            undefined, // Removed date parameter
            rowsPerPage,
            page * rowsPerPage,
            sortOrder,
            predictionStartDate || undefined,
            predictionEndDate || undefined
          );
          
          // Die Sortierung erfolgt jetzt auf Server-Ebene durch den sort_order Parameter
          setImages(response.images);
          setTotalImages(response.total);
        } else {
          // For "All Devices" option - we need to fetch for each device with images
          const deviceIdsWithImages = await getPredictionImageDevices();
          
          // Define the number of images to fetch per device to avoid overloading
          // Based on rowsPerPage, we'll try to get a fair distribution
          const imagesPerDevice = Math.max(Math.ceil(rowsPerPage / (deviceIdsWithImages.length || 1)), 5);
          
          // Fetch prediction images from all devices concurrently for better performance
          const imagePromises = deviceIdsWithImages.map(deviceId => 
            getPredictionImages(
              deviceId,
              selectedType || undefined,
              undefined, // Removed date parameter
              imagesPerDevice, // Get a few images from each device
              0, // For simplicity, just get the first page from each device
              sortOrder,
              predictionStartDate || undefined,
              predictionEndDate || undefined
            )
          );
          
          const allResults = await Promise.all(imagePromises);
          
          // Combine all images
          let allImages: PredictionImage[] = allResults.flatMap(response => response.images);
          
          // Apply date filtering on client side if needed
          if (predictionStartDate || predictionEndDate) {
            const startTimestamp = predictionStartDate ? new Date(predictionStartDate).getTime() : 0;
            const endTimestamp = predictionEndDate ? new Date(predictionEndDate + 'T23:59:59').getTime() : Infinity;
            
            allImages = allImages.filter(image => {
              const imageTimestamp = new Date(image.last_modified).getTime();
              return imageTimestamp >= startTimestamp && imageTimestamp <= endTimestamp;
            });
          }
          
          console.log(`Fetched ${allImages.length} prediction images from ${deviceIdsWithImages.length} devices`);
          
          // Sortieren der kombinierten Bilder von allen Geräten
          allImages.sort((a, b) => {
            const dateA = new Date(a.last_modified).getTime();
            const dateB = new Date(b.last_modified).getTime();
            
            return sortOrder === 'desc' 
              ? dateB - dateA  // Newest first (desc)
              : dateA - dateB; // Oldest first (asc)
          });
          
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
  }, [selectedDeviceId, selectedType, page, rowsPerPage, tabValue, sortOrder, predictionStartDate, predictionEndDate]);

  // Load operational logs based on filters and pagination
  useEffect(() => {
    const loadOperationalLogs = async () => {
      // Only load logs on the Logs tab
      if (tabValue !== 2) return;

      setLoadingLogs(true);
      setLogsError(null);

      try {
        if (selectedDeviceId && selectedDeviceId !== 'all') {
          // For a specific device
          const response = await getOperationalLogs(
            selectedDeviceId,
            selectedLogType || undefined,
            rowsPerPage,
            page * rowsPerPage,
            sortOrder,
            startDate || undefined,
            endDate || undefined
          );
          
          setOperationalLogs(response.logs);
          setTotalLogs(response.total);
        } else {
          // For "All Devices" option
          const deviceIdsWithLogs = await getOperationalLogDevices();
          
          // Define the number of logs to fetch per device to avoid overloading
          // Based on rowsPerPage, we'll try to get a fair distribution
          const logsPerDevice = Math.max(Math.ceil(rowsPerPage / (deviceIdsWithLogs.length || 1)), 5);
          
          // Fetch operational logs from all devices concurrently for better performance
          const logPromises = deviceIdsWithLogs.map(deviceId => 
            getOperationalLogs(
              deviceId,
              selectedLogType || undefined,
              logsPerDevice, // Get a few logs from each device
              0, // For simplicity, just get the first page from each device
              sortOrder,
              startDate || undefined,
              endDate || undefined
            )
          );
          
          const allResults = await Promise.all(logPromises);
          
          // Combine all logs
          let allLogs: OperationalLog[] = allResults.flatMap(response => response.logs);
          
          console.log(`Fetched ${allLogs.length} operational logs from ${deviceIdsWithLogs.length} devices`);
          
          // Apply date filtering on client side if needed
          if (startDate || endDate) {
            const startTimestamp = startDate ? new Date(startDate).getTime() : 0;
            const endTimestamp = endDate ? new Date(endDate + 'T23:59:59').getTime() : Infinity;
            
            allLogs = allLogs.filter(log => {
              const logTimestamp = new Date(log.last_modified).getTime();
              return logTimestamp >= startTimestamp && logTimestamp <= endTimestamp;
            });
          }
          
          // Sort all logs
          allLogs.sort((a, b) => {
            const dateA = new Date(a.last_modified).getTime();
            const dateB = new Date(b.last_modified).getTime();
            
            return sortOrder === 'desc' 
              ? dateB - dateA  // Newest first (desc)
              : dateA - dateB; // Oldest first (asc)
          });
          
          // Simple client-side pagination
          const startIndex = page * rowsPerPage;
          const endIndex = startIndex + rowsPerPage;
          
          setTotalLogs(allLogs.length);
          setOperationalLogs(allLogs.slice(startIndex, endIndex));
        }
      } catch (err) {
        console.error('Error loading operational logs:', err);
        setLogsError('Failed to load operational logs');
        setOperationalLogs([]);
        setTotalLogs(0);
      } finally {
        setLoadingLogs(false);
      }
    };
    
    loadOperationalLogs();
  }, [selectedDeviceId, selectedLogType, page, rowsPerPage, tabValue, sortOrder, startDate, endDate]);

  // Load drift images based on filters and pagination
  useEffect(() => {
    const loadDriftImages = async () => {
      // Only load drift images on the Drift Events tab
      if (tabValue !== 1) return;

      setLoadingDriftImages(true);
      setDriftError(null);

      try {
        if (selectedDeviceId && selectedDeviceId !== 'all') {
          // For a specific device
          const response = await getDriftImages(
            selectedDeviceId,
            selectedDriftType || undefined,
            undefined, // Removed date parameter
            rowsPerPage,
            page * rowsPerPage,
            sortOrder,
            driftStartDate || undefined,
            driftEndDate || undefined
          );

          // Process images to set event_id from folder name if available
          const processedImages = response.images.map(image => {
            // Check if the image is in a folder that starts with "event_"
            const pathParts = image.key.split('/');
            if (pathParts.length > 3) {
              const potentialEvent = pathParts[3]; // Check the 4th path component
              if (potentialEvent.startsWith('event_')) {
                // Extract event ID from folder name
                const eventId = potentialEvent.replace('event_', '');
                return {
                  ...image,
                  event_id: eventId
                };
              }
            }
            return image;
          });

          // Die Sortierung erfolgt jetzt auf Server-Ebene durch den sort_order Parameter
          setDriftImages(processedImages);
          setTotalDriftImages(response.total);
        } else {
          // For "All Devices" option - we need to fetch for each device with images
          const deviceIdsWithDriftImages = await getDriftImageDevices();

          // Define the number of images to fetch per device to avoid overloading
          // Based on rowsPerPage, we'll try to get a fair distribution
          const imagesPerDevice = Math.max(Math.ceil(rowsPerPage / (deviceIdsWithDriftImages.length || 1)), 5);
          
          // Fetch drift images from all devices concurrently for better performance
          const imagePromises = deviceIdsWithDriftImages.map(deviceId => 
            getDriftImages(
              deviceId,
              selectedDriftType || undefined,
              undefined, // Removed date parameter
              imagesPerDevice, // Get a few images from each device
              0, // For simplicity, just get the first page from each device
              sortOrder,
              driftStartDate || undefined,
              driftEndDate || undefined
            )
          );
          
          const allResults = await Promise.all(imagePromises);
          
          // Process and combine all drift images
          let allDriftImages: DriftImage[] = [];
          
          for (const response of allResults) {
            // Process images to set event_id from folder name if available
            const processedImages = response.images.map(image => {
              // Check if the image is in a folder that starts with "event_"
              const pathParts = image.key.split('/');
              if (pathParts.length > 3) {
                const potentialEvent = pathParts[3]; // Check the 4th path component
                if (potentialEvent.startsWith('event_')) {
                  // Extract event ID from folder name
                  const eventId = potentialEvent.replace('event_', '');
                  return {
                    ...image,
                    event_id: eventId
                  };
                }
              }
              return image;
            });

            allDriftImages = [...allDriftImages, ...processedImages];
          }

          // Apply date filtering on client side if needed
          if (driftStartDate || driftEndDate) {
            const startTimestamp = driftStartDate ? new Date(driftStartDate).getTime() : 0;
            const endTimestamp = driftEndDate ? new Date(driftEndDate + 'T23:59:59').getTime() : Infinity;
            
            allDriftImages = allDriftImages.filter(image => {
              const imageTimestamp = new Date(image.last_modified).getTime();
              return imageTimestamp >= startTimestamp && imageTimestamp <= endTimestamp;
            });
          }

          console.log(`Fetched ${allDriftImages.length} drift images from ${deviceIdsWithDriftImages.length} devices`);

          // Sortieren der kombinierten Drift-Bilder von allen Geräten
          allDriftImages.sort((a, b) => {
            const dateA = new Date(a.last_modified).getTime();
            const dateB = new Date(b.last_modified).getTime();
            
            return sortOrder === 'desc' 
              ? dateB - dateA  // Newest first (desc)
              : dateA - dateB; // Oldest first (asc)
          });

          // Simple client-side pagination
          const startIndex = page * rowsPerPage;
          const endIndex = startIndex + rowsPerPage;

          setTotalDriftImages(allDriftImages.length);
          setDriftImages(allDriftImages.slice(startIndex, endIndex));
        }
      } catch (err) {
        console.error('Error loading drift images:', err);
        setDriftError('Failed to load drift images');
        setDriftImages([]);
        setTotalDriftImages(0);
      } finally {
        setLoadingDriftImages(false);
      }
    };

    loadDriftImages();
  }, [selectedDeviceId, selectedDriftType, page, rowsPerPage, tabValue, sortOrder, driftStartDate, driftEndDate]);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Helper to convert DriftImage to PredictionImage for compatibility
  const convertDriftToPredictionImage = (image: DriftImage): PredictionImage => {
    return {
      ...image,
      prediction_type: image.drift_type
    } as PredictionImage;
  };
  
  // Handle device selection
  const handleDeviceChange = (event: SelectChangeEvent<string>) => {
    const deviceId = event.target.value;
    setSelectedDeviceId(deviceId);
    setSelectedType('');
    setPage(0);
  };
  
  // Handle prediction type selection
  const handleTypeChange = (event: SelectChangeEvent<string>) => {
    const type = event.target.value;
    setSelectedType(type);
    setPage(0);
  };
  
  // Date selection removed
  
  // Handle prediction start/end date changes
  const handlePredictionStartDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPredictionStartDate(event.target.value);
    setPage(0);
  };
  
  const handlePredictionEndDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPredictionEndDate(event.target.value);
    setPage(0);
  };

  // Handle drift type selection
  const handleDriftTypeChange = (event: SelectChangeEvent<string>) => {
    const type = event.target.value;
    setSelectedDriftType(type);
    setPage(0);
  };
  
  // Drift date selection removed
  
  // Handle drift start/end date changes
  const handleDriftStartDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setDriftStartDate(event.target.value);
    setPage(0);
  };
  
  const handleDriftEndDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setDriftEndDate(event.target.value);
    setPage(0);
  };
  
  // Handle log type selection for operational logs
  const handleLogTypeChange = (event: SelectChangeEvent<string>) => {
    const type = event.target.value;
    setSelectedLogType(type);
    setPage(0);
  };
  
  // Handle start date change for logs
  const handleStartDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setStartDate(event.target.value);
    setPage(0);
  };
  
  // Handle end date change for logs
  const handleEndDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEndDate(event.target.value);
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
  
  // Toggle sort order
  const handleToggleSortOrder = () => {
    setSortOrder(prevOrder => prevOrder === 'asc' ? 'desc' : 'asc');
    setPage(0); // Reset to first page when sorting changes
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
  
  // Open delete confirmation dialog for device logs
  const handleDeleteDeviceLogs = (deviceId: string) => {
    setDeleteType('device');
    setLogToDelete({ deviceId });
    setDeleteDialogOpen(true);
  };
  
  // Open delete confirmation dialog for session logs
  const handleDeleteSessionLogs = (deviceId: string, sessionId: string) => {
    setDeleteType('session');
    setLogToDelete({ deviceId, sessionId });
    setDeleteDialogOpen(true);
  };
  
  // Handle log deletion confirmation
  const handleConfirmDelete = async () => {
    if (!logToDelete) return;
    
    setIsDeleting(true);
    try {
      if (deleteType === 'device') {
        await deleteDeviceLogs(logToDelete.deviceId);
      } else if (deleteType === 'session' && logToDelete.sessionId) {
        await deleteSessionLogs(logToDelete.deviceId, logToDelete.sessionId);
      }
      
      // Refresh logs list
      setLoadingLogs(true);
      
      // Show success notification
      // This could be replaced with a proper notification system
      alert(`Successfully deleted ${deleteType === 'device' ? 'all device' : 'session'} logs`);
      
    } catch (error) {
      console.error('Error deleting logs:', error);
      alert(`Failed to delete logs: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsDeleting(false);
      setDeleteDialogOpen(false);
      setLogToDelete(null);
    }
  };
  
  // Close delete confirmation dialog
  const handleCancelDelete = () => {
    setDeleteDialogOpen(false);
    setLogToDelete(null);
    setIsDeleting(false);
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
  
  // Render drift images gallery
  const renderDriftImageGallery = () => {
    if (loadingDriftImages) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (driftError) {
      return (
        <ErrorDisplay 
          error={driftError}
          loading={false}
          onRetry={() => setLoadingDriftImages(true)}
          height="50vh"
        />
      );
    }
    
    if (driftImages.length === 0) {
      return (
        <Box sx={{ textAlign: 'center', p: 4 }}>
          <Typography variant="body1" color="text.secondary">
            No drift images found for the selected filters.
          </Typography>
          {!selectedDeviceId && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Please select a device to view available drift images.
            </Typography>
          )}
        </Box>
      );
    }
    
    return (
      <Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {driftImages.map((image) => {
            // Convert drift image to prediction image for compatibility
            const compatImage = convertDriftToPredictionImage(image);
            
            return (
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
                  onClick={() => handleImageClick(compatImage)}
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
                        label={image.drift_type} 
                        size="small" 
                        color="warning"
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
                    {image.event_id && (
                      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 0.5 }}>
                        <Button
                          size="small"
                          color="primary"
                          component="a"
                          href={`/drift/${image.event_id}`}
                          sx={{ fontSize: '0.7rem', py: 0, px: 1 }}
                        >
                          View Event
                        </Button>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Box>
            );
          })}
        </Box>
        
        {/* Pagination */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <TablePagination
            component="div"
            count={totalDriftImages}
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
  
  // Render prediction image filters
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
          <TextField
            label="Start Date"
            type="date"
            value={predictionStartDate}
            onChange={handlePredictionStartDateChange}
            fullWidth
            size="small"
            InputLabelProps={{ shrink: true }}
          />
        </Box>
        
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <TextField
            label="End Date"
            type="date"
            value={predictionEndDate}
            onChange={handlePredictionEndDateChange}
            fullWidth
            size="small"
            InputLabelProps={{ shrink: true }}
          />
        </Box>
        
        <Box sx={{ flex: '1 1 200px', display: 'flex', alignItems: 'center', gap: 1 }}>
          <Button 
            variant="outlined" 
            disabled={!selectedType && !predictionStartDate && !predictionEndDate}
            onClick={() => {
              setSelectedType('');
              setPredictionStartDate('');
              setPredictionEndDate('');
            }}
          >
            Clear Filters
          </Button>
          <Tooltip title={sortOrder === 'desc' ? "Showing newest first - Click to show oldest first" : "Showing oldest first - Click to show newest first"}>
            <IconButton onClick={handleToggleSortOrder} color="primary">
              {sortOrder === 'desc' ? <SortDescIcon /> : <SortAscIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Box>
  );

  // Render drift image filters
  const renderDriftImageFilters = () => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <FormControl fullWidth size="small" disabled={!selectedDeviceId || driftTypes.length === 0}>
            <InputLabel id="drift-type-select-label">Drift Type</InputLabel>
            <Select
              labelId="drift-type-select-label"
              value={selectedDriftType}
              label="Drift Type"
              onChange={handleDriftTypeChange}
            >
              {driftTypes.map(type => (
                <MenuItem key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <TextField
            label="Start Date"
            type="date"
            value={driftStartDate}
            onChange={handleDriftStartDateChange}
            fullWidth
            size="small"
            InputLabelProps={{ shrink: true }}
          />
        </Box>
        
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <TextField
            label="End Date"
            type="date"
            value={driftEndDate}
            onChange={handleDriftEndDateChange}
            fullWidth
            size="small"
            InputLabelProps={{ shrink: true }}
          />
        </Box>
        
        <Box sx={{ flex: '1 1 200px', display: 'flex', alignItems: 'center', gap: 1 }}>
          <Button 
            variant="outlined" 
            disabled={!selectedDriftType && !driftStartDate && !driftEndDate}
            onClick={() => {
              setSelectedDriftType('');
              setDriftStartDate('');
              setDriftEndDate('');
            }}
          >
            Clear Filters
          </Button>
          <Tooltip title={sortOrder === 'desc' ? "Showing newest first - Click to show oldest first" : "Showing oldest first - Click to show newest first"}>
            <IconButton onClick={handleToggleSortOrder} color="primary">
              {sortOrder === 'desc' ? <SortDescIcon /> : <SortAscIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Box>
  );
  
  // Render log type selector
  const renderLogTypeSelector = () => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <FormControl fullWidth size="small" disabled={!selectedDeviceId || logTypes.length === 0}>
            <InputLabel id="log-type-select-label">Log Type</InputLabel>
            <Select
              labelId="log-type-select-label"
              value={selectedLogType}
              label="Log Type"
              onChange={handleLogTypeChange}
            >
              {logTypes.map(type => (
                <MenuItem key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <TextField
            label="Start Date"
            type="date"
            value={startDate}
            onChange={handleStartDateChange}
            fullWidth
            size="small"
            InputLabelProps={{ shrink: true }}
          />
        </Box>
        
        <Box sx={{ flex: '1 1 200px', minWidth: '150px' }}>
          <TextField
            label="End Date"
            type="date"
            value={endDate}
            onChange={handleEndDateChange}
            fullWidth
            size="small"
            InputLabelProps={{ shrink: true }}
          />
        </Box>
        
        <Box sx={{ flex: '1 1 200px', display: 'flex', alignItems: 'center', gap: 1 }}>
          <Button 
            variant="outlined" 
            disabled={!selectedLogType && !startDate && !endDate}
            onClick={() => {
              setSelectedLogType('');
              setStartDate('');
              setEndDate('');
            }}
          >
            Clear Filters
          </Button>
          <Tooltip title={sortOrder === 'desc' ? "Showing newest first - Click to show oldest first" : "Showing oldest first - Click to show newest first"}>
            <IconButton onClick={handleToggleSortOrder} color="primary">
              {sortOrder === 'desc' ? <SortDescIcon /> : <SortAscIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Box>
  );
  
  // Render operational logs table
  const renderOperationalLogs = () => {
    if (loadingLogs) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (logsError) {
      return (
        <ErrorDisplay 
          error={logsError}
          loading={false}
          onRetry={() => setLoadingLogs(true)}
          height="50vh"
        />
      );
    }
    
    if (operationalLogs.length === 0) {
      return (
        <Box sx={{ textAlign: 'center', p: 4 }}>
          <Typography variant="body1" color="text.secondary">
            No operational logs found for the selected filters.
          </Typography>
          {!selectedDeviceId && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Please select a device to view available logs.
            </Typography>
          )}
        </Box>
      );
    }
    
    // Convert bytes to human-readable format
    const formatBytes = (bytes: number): string => {
      if (bytes === 0) return '0 Bytes';
      
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(1024));
      
      return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    // Format the timestamp
    const formatTimestamp = (timestamp: string): string => {
      try {
        return format(parseISO(timestamp), 'MMM d, yyyy HH:mm:ss');
      } catch {
        return timestamp;
      }
    };
    
    return (
      <Box>
        <Paper elevation={2} sx={{ borderRadius: 1, overflow: 'hidden' }}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold' }}>File Name</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Device ID</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Session ID</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Size</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Last Modified</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {operationalLogs.map((log, index) => (
                  <TableRow 
                    key={log.key} 
                    hover
                    sx={{ 
                      bgcolor: 'white',
                      '&:hover': { 
                        bgcolor: 'action.hover' 
                      }
                    }}
                  >
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LogFileIcon sx={{ color: 'primary.main' }} />
                        <Typography variant="body2">{log.filename}</Typography>
                      </Box>
                    </TableCell>
                    <TableCell>{log.device_id}</TableCell>
                    <TableCell>{log.session_id}</TableCell>
                    <TableCell>{formatBytes(log.size)}</TableCell>
                    <TableCell>{formatTimestamp(log.last_modified)}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Delete Log">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteSessionLogs(log.device_id, log.session_id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Download">
                          <IconButton
                            size="small"
                            onClick={() => window.location.href = log.url}
                          >
                            <DownloadIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Pagination */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <TablePagination
            component="div"
            count={totalLogs}
            page={page}
            onPageChange={handleChangePage}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={handleChangeRowsPerPage}
            rowsPerPageOptions={[10, 25, 50, 100]}
          />
        </Box>
      </Box>
    );
  };
  
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
              icon={<GalleryIcon />}
              iconPosition="start"
              label="Prediction Images"
              disabled={!selectedDeviceId}
            />
            <Tab
              icon={<WarningAmber sx={{ color: 'orange' }} />}
              iconPosition="start"
              label="Drift Events"
              disabled={!selectedDeviceId}
            />
            <Tab
              icon={<ListIcon />}
              iconPosition="start"
              label="Logs"
              disabled={!selectedDeviceId}
            />
          </Tabs>
        </Box>
      </Paper>
      
      {/* Tab content */}
      <TabPanel value={tabValue} index={0}>
        {/* Filters for prediction images */}
        {selectedDeviceId && renderImageFilters()}

        {/* Images gallery */}
        {renderImageGallery()}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {/* Filters for drift images */}
        {selectedDeviceId && renderDriftImageFilters()}

        {/* Drift images gallery */}
        {renderDriftImageGallery()}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {/* Filters for operational logs */}
        {selectedDeviceId && renderLogTypeSelector()}

        {/* Operational logs table */}
        {renderOperationalLogs()}
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
      
      {/* Log deletion confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleCancelDelete}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Confirm Log Deletion
        </DialogTitle>
        <DialogContent>
          <Typography>
            {deleteType === 'device' 
              ? `Are you sure you want to delete all logs for device ${logToDelete?.deviceId}?` 
              : `Are you sure you want to delete all logs for session ${logToDelete?.sessionId} on device ${logToDelete?.deviceId}?`
            }
          </Typography>
          <Typography color="error" sx={{ mt: 2 }}>
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 2, gap: 1 }}>
          <Button 
            onClick={handleCancelDelete}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button 
            variant="contained" 
            color="error"
            onClick={handleConfirmDelete}
            disabled={isDeleting}
            startIcon={isDeleting ? <CircularProgress size={20} /> : <DeleteIcon />}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </Box>
      </Dialog>
    </Box>
  );
};

export default DataHub;