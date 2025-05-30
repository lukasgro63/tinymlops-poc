// src/pages/Dashboard.tsx
import InventoryIcon from '@mui/icons-material/Inventory';
import MemoryIcon from '@mui/icons-material/Memory';
import NotificationsIcon from '@mui/icons-material/Notifications';
import PsychologyIcon from '@mui/icons-material/Psychology';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import StorageIcon from '@mui/icons-material/Storage';
import TimelineIcon from '@mui/icons-material/Timeline';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import FilterListIcon from '@mui/icons-material/FilterList';
import RefreshIcon from '@mui/icons-material/Refresh';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import { Box, Typography, Tooltip, IconButton } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorDisplay from '../components/common/ErrorDisplay';

// Components
import ModelPerformanceChart from '../components/common/ModelPerformanceChart';
import DeviceMapChart from '../components/common/DeviceMapChart';
import PackageUploadChart from '../components/common/PackageUploadChart';
import RecentActivitiesList from '../components/common/RecentActivitiesList';
import RecentDriftEventsTable from '../components/common/RecentDriftEventsTable';
import SectionCard from '../components/common/SectionCard';
import StatusCard from '../components/common/StatusCard';
import SystemStatusIndicator from '../components/common/SystemStatusIndicator';

// API Services
import {
  getDevicesSummary,
  getDeviceLocations,
  getModels,
  getModelsPerformance,
  getModelsSummary,
  getPackageActivity,
  getPackageTimeline,
  getSystemStatus
} from '../services/api';
import {
  DeviceSummary,
  DeviceLocation,
  DeviceLocationsResponse,
  ModelPerformanceData,
  ModelSummary,
  PackageActivity,
  PackageTimelineData,
} from '../types/api';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  // State for dashboard data
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [deviceSummaries, setDeviceSummaries] = useState<DeviceSummary[]>([]);
  const [deviceLocations, setDeviceLocations] = useState<DeviceLocation[]>([]);
  const [modelSummaries, setModelSummaries] = useState<ModelSummary[]>([]);
  const [packageActivities, setPackageActivities] = useState<PackageActivity[]>([]);
  const [modelNames, setModelNames] = useState<string[]>([]);
  const [timelineData, setTimelineData] = useState<PackageTimelineData[]>([]);
  const [performanceData, setPerformanceData] = useState<ModelPerformanceData[]>([]);
  
  // Selected time period for package timeline
  const [timelinePeriod, setTimelinePeriod] = useState<'day' | 'week' | 'month'>('week');
  
  // Loading and error states
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Model Performance Chart state
  const [modelPerformanceFiltersVisible, setModelPerformanceFiltersVisible] = useState<boolean>(true);
  const [modelPerformanceLastUpdated, setModelPerformanceLastUpdated] = useState<Date | null>(null);
  const modelPerformanceChartRef = React.useRef<any>(null);
  
  // Fetch all dashboard data on mount
  useEffect(() => {
    fetchDashboardData();
    
    // Set up polling interval (every 30 seconds)
    const interval = setInterval(() => {
      fetchDashboardData();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Fetch package timeline data when period changes
  useEffect(() => {
    fetchPackageTimeline(timelinePeriod);
  }, [timelinePeriod]);
  
  // Function to fetch all dashboard data
  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch all data in parallel
      const [
        statusData,
        devicesData,
        locationsData,
        modelsData,
        activitiesData,
        modelNamesData,
        performanceMetricsData
      ] = await Promise.all([
        getSystemStatus(),
        getDevicesSummary(),
        getDeviceLocations(),
        getModelsSummary(),
        getPackageActivity(7),
        getModels(),
        getModelsPerformance('accuracy')
      ]);
      
      // Update all state
      setSystemStatus(statusData);
      setDeviceSummaries(devicesData);
      setDeviceLocations(locationsData.locations || []);
      setModelSummaries(modelsData);
      setPackageActivities(activitiesData);
      setModelNames(modelNamesData);
      setPerformanceData(performanceMetricsData);
      
      // Fetch package timeline separately (depends on timelinePeriod)
      await fetchPackageTimeline(timelinePeriod);
      
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError('Failed to load dashboard data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };
  
  // Function to fetch package timeline data
  const fetchPackageTimeline = async (period: 'day' | 'week' | 'month') => {
    try {
      const data = await getPackageTimeline(period);
      setTimelineData(data);
    } catch (error) {
      console.error('Error fetching package timeline:', error);
      // Don't set global error - this is just one component
    }
  };
  
  
  // Show loading or error state if applicable
  if ((loading && !systemStatus) || (error && !systemStatus)) {
    return (
      <ErrorDisplay 
        error={error}
        loading={loading && !systemStatus}
        onRetry={fetchDashboardData}
        height="100vh"
      />
    );
  }
  
  // Get models in production count
  const modelsInProduction = modelSummaries?.filter(m => m.has_production).length || 0;
  
  return (
    <Box>
      {/* System Status */}
      {systemStatus && (
        <SystemStatusIndicator status={systemStatus.status} />
      )}
      
      {/* Top Row: Status Cards */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
        {/* Devices Card */}
        <Box sx={{ flex: '1 1 220px', minWidth: '220px' }}>
          <StatusCard
            title="Devices"
            value={systemStatus?.statistics.active_devices || 0}
            secondaryValue={`${systemStatus?.statistics.total_devices || 0}`}
            secondaryLabel="Total"
            icon={<MemoryIcon style={{ fontSize: 24, color: '#00647D' }} />}
            color="#00647D" // TinyLCM primary color
          />
        </Box>
        
        {/* Models Card */}
        <Box sx={{ flex: '1 1 220px', minWidth: '220px' }}>
          <StatusCard
            title="Models"
            value={modelSummaries?.length || 0}
            secondaryValue={`${modelsInProduction}`}
            secondaryLabel="In Production"
            icon={<PsychologyIcon style={{ fontSize: 24, color: '#E5A823' }} />}
            color="#E5A823" // TinyLCM secondary color
          />
        </Box>
        
        {/* Packages Card */}
        <Box sx={{ flex: '1 1 220px', minWidth: '220px' }}>
          <StatusCard
            title="Packages"
            value={`${systemStatus?.statistics.processed_packages || 0}/${systemStatus?.statistics.total_packages || 0}`}
            secondaryLabel="Processed/Total"
            icon={<InventoryIcon style={{ fontSize: 24, color: '#4CAF50' }} />}
            color="#4CAF50" // Success color
          />
        </Box>
        
        {/* MLflow Card */}
        <Box sx={{ flex: '1 1 220px', minWidth: '220px' }}>
          <StatusCard
            title="MLflow"
            value="Connected"
            secondaryValue={`${modelNames.length || 0} Models`}
            icon={<StorageIcon style={{ fontSize: 24, color: '#4CAF50' }} />}
            color="#4CAF50"
          />
        </Box>
      </Box>
      
      {/* Middle Row: Activities and Timeline */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
        {/* Recent Activities */}
        <Box sx={{ flex: '1 1 400px', minWidth: '300px' }}>
          <SectionCard
            title="Recent Activities"
            icon={<NotificationsIcon style={{ fontSize: 20, color: '#00647D' }} />}
            maxHeight={400}
          >
            <RecentActivitiesList activities={packageActivities} />
          </SectionCard>
        </Box>
        
        {/* Package Upload Timeline */}
        <Box sx={{ flex: '1 1 400px', minWidth: '300px' }}>
          <SectionCard 
            title="Package Upload Timeline"
            icon={<TimelineIcon style={{ fontSize: 20, color: '#00647D' }} />}
            height={400}
          >
            <PackageUploadChart 
              data={timelineData} 
              onTimeRangeChange={setTimelinePeriod}
              currentTimeRange={timelinePeriod}
            />
          </SectionCard>
        </Box>
      </Box>
      
      {/* Bottom Row: Model Performance and Device Map */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
        {/* Model Performance */}
        <Box sx={{ flex: '2 1 600px', minWidth: '500px' }}>
          <SectionCard 
            title="Model Performance Trends"
            icon={<ShowChartIcon style={{ fontSize: 20, color: '#00647D' }} />}
            height={400}
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="Last updated">
                  <Typography variant="caption" sx={{ alignSelf: 'center', mr: 1, color: 'text.secondary' }}>
                    {modelPerformanceLastUpdated ? `Updated: ${modelPerformanceLastUpdated.toLocaleTimeString()}` : ''}
                  </Typography>
                </Tooltip>
                <Tooltip title={modelPerformanceFiltersVisible ? "Hide Filters" : "Show Filters"}>
                  <IconButton 
                    size="small" 
                    onClick={() => setModelPerformanceFiltersVisible(!modelPerformanceFiltersVisible)}
                    color={modelPerformanceFiltersVisible ? "primary" : "default"}
                  >
                    <FilterListIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh Data">
                  <IconButton 
                    size="small" 
                    onClick={() => {
                      if (modelPerformanceChartRef.current?.refresh) {
                        modelPerformanceChartRef.current.refresh();
                      }
                      setModelPerformanceLastUpdated(new Date());
                    }}
                  >
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          >
            <ModelPerformanceChart 
              models={modelNames} 
              performanceData={performanceData}
              ref={modelPerformanceChartRef}
              filtersVisible={modelPerformanceFiltersVisible}
              onLastUpdated={(date) => setModelPerformanceLastUpdated(date)}
            />
          </SectionCard>
        </Box>

        {/* Device Map */}
        <Box sx={{ flex: '1 1 400px', minWidth: '300px' }}>
          <SectionCard 
            title="Device Locations"
            icon={<LocationOnIcon style={{ fontSize: 20, color: '#00647D' }} />}
            height={400}
          >
            <DeviceMapChart 
              devices={deviceLocations} 
              height={340} 
              onDeviceClick={(deviceId) => navigate(`/devices?device=${deviceId}`)}
            />
          </SectionCard>
        </Box>
      </Box>
      
      {/* Removed Recent Drift Events Table */}
    </Box>
  );
};

export default Dashboard;