// src/components/dashboard/RecentActivitiesList.tsx
import { Box, Divider, Typography } from '@mui/material';
import React from 'react';
import { PackageActivity } from '../../types/api';

interface RecentActivitiesListProps {
  activities: PackageActivity[];
}

const RecentActivitiesList: React.FC<RecentActivitiesListProps> = ({ activities }) => {
  if (!activities || activities.length === 0) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No recent activity found
        </Typography>
      </Box>
    );
  }

  // Format date to relative time (without date-fns dependency)
  const formatRelativeTime = (dateString: string) => {
    try {
      // Explicitly handle the Z suffix to ensure UTC parsing
      const dateStr = dateString.endsWith('Z') ? dateString : dateString + 'Z';
      
      // Parse the date string as UTC, then convert to local time
      const date = new Date(dateStr);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      
      // Convert to appropriate unit
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
      return 'unknown time';
    }
  };

  // Format date to absolute time
  const formatAbsoluteTime = (dateString: string) => {
    try {
      // Explicitly handle the Z suffix to ensure UTC parsing
      const dateStr = dateString.endsWith('Z') ? dateString : dateString + 'Z';
      const date = new Date(dateStr);
      return date.toLocaleString();
    } catch (e) {
      console.error("Date parsing error:", e, "for date:", dateString);
      return 'unknown time';
    }
  };

  return (
    <Box>
      {activities.map((activity, index) => (
        <React.Fragment key={activity.package_id}>
          <Box 
            sx={{ 
              px: 3, 
              py: 2,
              cursor: 'pointer',
              transition: 'background-color 0.2s',
              '&:hover': { 
                backgroundColor: 'rgba(0, 0, 0, 0.04)' 
              }
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body1">
                {activity.package_type} package
              </Typography>
              <Box 
                sx={{ 
                  px: 1.5, 
                  py: 0.5, 
                  borderRadius: '4px', 
                  fontSize: '0.75rem',
                  fontWeight: 'medium',
                  backgroundColor: 
                    activity.processing_status === 'success' ? 'rgba(76, 175, 80, 0.1)' :
                    activity.processing_status === 'error' ? 'rgba(244, 67, 54, 0.1)' :
                    'rgba(255, 152, 0, 0.1)',
                  color:
                    activity.processing_status === 'success' ? '#4CAF50' :
                    activity.processing_status === 'error' ? '#F44336' :
                    '#FF9800'
                }}
              >
                {activity.processing_status}
              </Box>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              From {activity.device_name || activity.device_id} • {formatRelativeTime(activity.uploaded_at)}
            </Typography>
          </Box>
          {index < activities.length - 1 && <Divider />}
        </React.Fragment>
      ))}
    </Box>
  );
};

export default RecentActivitiesList;