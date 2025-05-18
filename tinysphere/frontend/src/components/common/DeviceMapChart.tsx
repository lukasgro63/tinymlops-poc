import { Box, Typography, useTheme } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { Device, DeviceLocation } from '../../types/api';

// Fix for Leaflet marker icons in webpack builds
// This is needed because Leaflet expects the marker images to be in the same directory as the CSS
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

interface DeviceMapChartProps {
  devices: DeviceLocation[];
  height?: string | number;
  onDeviceClick?: (deviceId: string) => void;
}

const DEFAULT_CENTER: [number, number] = [40, 0]; // Default center if no devices
const DEFAULT_ZOOM = 2;

const DeviceMapChart: React.FC<DeviceMapChartProps> = ({ 
  devices, 
  height = 400,
  onDeviceClick 
}) => {
  const theme = useTheme();
  const [mapCenter, setMapCenter] = useState<[number, number]>(DEFAULT_CENTER);
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);
  
  // Calculate map center based on device locations
  useEffect(() => {
    if (!devices || devices.length === 0) return;
    
    // Filter out devices with invalid coordinates
    const validDevices = devices.filter(
      device => typeof device.latitude === 'number' && 
                typeof device.longitude === 'number' &&
                !isNaN(device.latitude) && 
                !isNaN(device.longitude)
    );
    
    if (validDevices.length === 0) return;
    
    // Calculate the average lat/lng of all devices to center the map
    const totalLat = validDevices.reduce((sum, device) => sum + device.latitude, 0);
    const totalLng = validDevices.reduce((sum, device) => sum + device.longitude, 0);
    
    if (totalLat && totalLng) {
      setMapCenter([totalLat / validDevices.length, totalLng / validDevices.length]);
      // Adjust zoom level if we have just one device
      setZoom(validDevices.length === 1 ? 10 : DEFAULT_ZOOM);
    }
  }, [devices]);

  if (devices.length === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: height,
        flexDirection: 'column',
        color: 'text.secondary' 
      }}>
        <Typography variant="body1">No device location data available</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: height, width: '100%' }}>
      <MapContainer 
        center={mapCenter} 
        zoom={zoom} 
        style={{ height: '100%', width: '100%', borderRadius: theme.shape.borderRadius }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {devices
          .filter(device => 
            typeof device.latitude === 'number' && 
            typeof device.longitude === 'number' &&
            !isNaN(device.latitude) && 
            !isNaN(device.longitude)
          )
          .map((device) => (
            <Marker 
              key={device.device_id}
              position={[device.latitude, device.longitude]}
              eventHandlers={{
                click: () => {
                  if (onDeviceClick) {
                    onDeviceClick(device.device_id);
                  }
                }
              }}
            >
              {/* Popup erscheint beim Klicken */}
              <Popup>
                <Typography variant="subtitle1" fontWeight="medium">
                  {device.device_id}
                </Typography>
                {device.location_name && (
                  <Typography variant="body2">
                    Location: {device.location_name}
                  </Typography>
                )}
                <Typography variant="caption" display="block">
                  Last update: {new Date(device.last_update || Date.now()).toLocaleString()}
                </Typography>
              </Popup>
              
              {/* Tooltip erscheint beim Hover */}
              <Tooltip direction="top" offset={[0, -20]} opacity={1} permanent={false}>
                <div>
                  <strong>{device.device_id}</strong><br />
                  Last update: {new Date(device.last_update || Date.now()).toLocaleString()}
                </div>
              </Tooltip>
            </Marker>
          ))}
      </MapContainer>
    </Box>
  );
};

export default DeviceMapChart;