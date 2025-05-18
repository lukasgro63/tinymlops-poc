# Geolocation in TinyMLOps

## Overview

The geolocation feature allows devices to send their geographical location to the TinySphere backend, enabling better monitoring and management of deployed ML models based on their physical location. This provides valuable insights into how models perform across different geographical regions and environments.

## Features

- **WiFi-based Geolocation**: Primary method for determining device location without requiring dedicated GPS hardware
- **IP-based Fallback**: Secondary method used when WiFi-based location is unavailable
- **Location Caching**: Reduces unnecessary lookups and improves performance
- **Configurable Update Intervals**: Control how frequently location data is updated
- **Map Visualization**: Display devices on an interactive map in the UI with density indicators
- **Device Location Management**: View and edit device locations in the device details

## Configuration

The geolocation feature can be enabled or disabled through the application configuration file. Below is an example configuration section:

```json
"geolocation": {
  "enabled": true,
  "api_key": null,
  "cache_ttl": 86400,
  "update_interval_seconds": 3600,
  "fallback_coordinates": [0.0, 0.0]
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable or disable geolocation | `false` |
| `api_key` | API key for geolocation service provider (if required) | `null` |
| `cache_ttl` | Time-to-live for location cache in seconds | `86400` (24 hours) |
| `update_interval_seconds` | Interval between location updates in seconds | `3600` (1 hour) |
| `fallback_coordinates` | Default coordinates to use if geolocation fails | `[0.0, 0.0]` |

## How it Works

### Client-Side Implementation

1. **WiFi Network Scanning**: The device scans available WiFi networks to collect access point data (BSSID, SSID, signal strength)
2. **API Lookup**: This data is sent to a geolocation service API which returns location coordinates
3. **Caching**: Location data is cached locally to reduce API calls and improve performance
4. **Fallback Mechanism**: If WiFi-based geolocation fails, the system falls back to IP-based geolocation
5. **Periodic Updates**: Location data is updated based on the configured interval
6. **Data Transmission**: Current location is sent to TinySphere during the regular sync process

#### Location Source Priority

1. WiFi-based positioning (most accurate, ~10-20m)
2. IP-based geolocation (less accurate, ~city level)
3. Fallback coordinates (specified in config)

### Backend Implementation

1. **Data Storage**: The TinySphere backend stores device location data in the `devices` table
2. **API Endpoints**: Dedicated endpoints for retrieving device locations 
3. **Location Management**: APIs for updating device locations manually

### Frontend Implementation

1. **Map Display**: Interactive map showing all devices with location data
2. **Clustering**: Devices in close proximity are clustered with size indicators
3. **Device Details**: Location information is displayed in the device details panel
4. **Location Editing**: Manual location update capability for administrators

## Technical Details

### WiFi-based Positioning

The implementation supports different WiFi scanning methods based on the operating system:

- **Linux**: Uses `iwlist` or similar tools to scan WiFi networks
- **macOS**: Uses `airport` utility to scan WiFi networks
- **Windows**: Uses native Windows API through Python libraries

### Dependencies

To use the geolocation feature, install the following dependencies:

```bash
# Core geolocation dependencies
pip install tinylcm[geolocation]

# For frontend map display
npm install leaflet react-leaflet @types/leaflet
```

## Privacy Considerations

When implementing geolocation tracking, several privacy considerations should be addressed:

1. **User Consent**: Ensure users are informed about location tracking
2. **Data Minimization**: Only collect location data at the frequency needed
3. **Data Security**: Ensure location data is securely stored and transmitted
4. **Retention Policies**: Implement appropriate data retention periods
5. **Access Controls**: Restrict access to location data to authorized users

## Known Limitations

1. **WiFi Dependency**: WiFi-based positioning requires sufficient WiFi access points nearby
2. **IP Geolocation Accuracy**: IP-based geolocation is generally only accurate to city level
3. **Indoor Positioning**: Accuracy may be reduced in indoor environments
4. **Rural Areas**: Less accurate in areas with few WiFi access points
5. **VPN/Proxies**: IP-based geolocation may be inaccurate when VPNs or proxies are used

## Troubleshooting

### Common Issues

1. **No Location Data**: 
   - Check if geolocation is enabled in configuration
   - Verify WiFi is enabled on the device
   - Check internet connectivity

2. **Inaccurate Location**:
   - Try increasing the sample size for WiFi scans
   - Update to the latest geolocation database
   - Consider using a dedicated GPS module for critical applications

3. **High Battery Usage**:
   - Increase the update interval
   - Use more aggressive caching

4. **Map Not Displaying**:
   - Check if the required frontend libraries are installed
   - Verify that the device has valid location data
   - Check browser console for errors

### Logs to Check

- Client logs: Look for "Geolocation" entries
- Backend logs: Check for device registration and update operations
- Frontend console: Check for map rendering issues

## Use Cases

1. **Regional Performance Analysis**: Compare model performance across different regions
2. **Deployment Planning**: Strategically deploy models to underserved areas
3. **Drift Detection by Region**: Identify potential drift patterns correlated with geographic regions
4. **Environmental Analysis**: Correlate model performance with environmental factors in different locations
5. **Maintenance Planning**: Optimize maintenance schedules based on device locations