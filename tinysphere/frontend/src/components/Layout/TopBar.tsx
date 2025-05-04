// src/components/Layout/TopBar.tsx
import CloseIcon from '@mui/icons-material/Close';
import MenuIcon from '@mui/icons-material/Menu';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import NotificationsIcon from '@mui/icons-material/Notifications';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  AppBar,
  Badge,
  Box,
  Breadcrumbs,
  Divider,
  IconButton,
  LinearProgress,
  Menu,
  MenuItem,
  Link as MuiLink,
  Toolbar,
  Tooltip,
  Typography,
  useTheme
} from '@mui/material';
import React, { useEffect, useState, useCallback } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  getNotifications, 
  getNotificationCounts, 
  markNotificationAsRead, 
  markAllNotificationsAsRead, 
  dismissNotification 
} from '../../services/api';
import { Notification } from '../../types/api';

interface TopBarProps {
  title: string; // Keep this prop for compatibility with existing code
  onMenuClick?: () => void;
  onRefresh?: () => void;
  sidebarOpen?: boolean;
}

// Breadcrumb mapping - maps routes to human-readable names
const pathMappings: Record<string, string> = {
  '': 'Home',
  'devices': 'Devices',
  'models': 'Models',
  'mlflow': 'MLflow',
  'grafana': 'Grafana',
  'minio': 'MinIO',
  'postgres': 'PostgreSQL',
  'documentation': 'Documentation',
  'api-docs': 'API Documentation'
};

// Helper to format dates
const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return `${date.toLocaleTimeString()} - ${date.toLocaleDateString()}`;
};

const TopBar: React.FC<TopBarProps> = ({ 
  title, // Keep this prop even though we use breadcrumbs instead
  onMenuClick, 
  onRefresh,
  sidebarOpen = true 
}) => {
  const theme = useTheme();
  const location = useLocation();
  const [notificationsAnchor, setNotificationsAnchor] = useState<null | HTMLElement>(null);
  
  // State for notifications
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Fetch notifications
  const fetchNotifications = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await getNotifications(0, 50, false);
      setNotifications(response.items);
      setUnreadCount(response.unread);
    } catch (err) {
      console.error('Failed to fetch notifications:', err);
      setError('Failed to load notifications');
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Fetch unread count only (for badge)
  const fetchUnreadCount = useCallback(async () => {
    try {
      const counts = await getNotificationCounts();
      setUnreadCount(counts.unread);
    } catch (err) {
      console.error('Failed to fetch notification counts:', err);
    }
  }, []);
  
  // Initial fetch
  useEffect(() => {
    fetchUnreadCount();
    
    // Set up polling for notifications count (every 30 seconds)
    const intervalId = setInterval(fetchUnreadCount, 30000);
    
    return () => clearInterval(intervalId);
  }, [fetchUnreadCount]);
  
  // Generate breadcrumbs from current path
  const generateBreadcrumbs = () => {
    const pathParts = location.pathname.split('/').filter(Boolean);
    
    // Set colors based on sidebar state
    const textColor = sidebarOpen ? 'white' : 'primary.main';
    const separatorColor = sidebarOpen ? 'white' : 'primary.main';
    
    // If we're on the home page, just show "Home"
    if (pathParts.length === 0) {
      return (
        <Breadcrumbs 
          separator={<NavigateNextIcon fontSize="small" sx={{ color: separatorColor }} />} 
          aria-label="breadcrumb"
        >
          <Typography color={textColor}>Home</Typography>
        </Breadcrumbs>
      );
    }
    
    // Create an array of breadcrumb items
    return (
      <Breadcrumbs 
        separator={<NavigateNextIcon fontSize="small" sx={{ color: separatorColor }} />}
        aria-label="breadcrumb"
      >
        <MuiLink 
          component={Link} 
          to="/" 
          color={textColor} 
          underline="hover"
          sx={{ fontWeight: 'normal' }}
        >
          Home
        </MuiLink>
        
        {pathParts.map((part, index) => {
          const path = `/${pathParts.slice(0, index + 1).join('/')}`;
          const isLast = index === pathParts.length - 1;
          
          // Get the human-readable name or capitalize the part
          const name = pathMappings[part] || part.charAt(0).toUpperCase() + part.slice(1);
          
          return isLast ? (
            <Typography key={path} color={textColor} fontWeight="medium">{name}</Typography>
          ) : (
            <MuiLink
              key={path}
              component={Link}
              to={path}
              color={textColor}
              underline="hover"
              sx={{ fontWeight: 'normal' }}
            >
              {name}
            </MuiLink>
          );
        })}
      </Breadcrumbs>
    );
  };
  
  const handleRefresh = () => {
    if (onRefresh) {
      onRefresh();
    } else {
      window.location.reload();
    }
  };
  
  const handleNotificationsOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationsAnchor(event.currentTarget);
    fetchNotifications();
  };
  
  const handleNotificationsClose = () => {
    setNotificationsAnchor(null);
  };
  
  const handleMarkAsRead = async (notificationId: number) => {
    try {
      await markNotificationAsRead(notificationId);
      
      // Update local state
      setNotifications(currentNotifications => 
        currentNotifications.map(notif => 
          notif.id === notificationId 
            ? { ...notif, read: true, read_at: new Date().toISOString() } 
            : notif
        )
      );
      
      // Update unread count
      setUnreadCount(prev => Math.max(0, prev - 1));
    } catch (err) {
      console.error('Failed to mark notification as read:', err);
    }
  };
  
  const handleDismissNotification = async (id: number, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent triggering the menu item click
    
    try {
      await dismissNotification(id);
      
      // Update local state
      setNotifications(currentNotifications => 
        currentNotifications.filter(notif => notif.id !== id)
      );
      
      // If it was unread, update the count
      const wasUnread = notifications.find(n => n.id === id)?.read === false;
      if (wasUnread) {
        setUnreadCount(prev => Math.max(0, prev - 1));
      }
    } catch (err) {
      console.error('Failed to dismiss notification:', err);
    }
  };
  
  const handleMarkAllAsRead = async () => {
    try {
      await markAllNotificationsAsRead();
      
      // Update local state
      setNotifications(currentNotifications => 
        currentNotifications.map(notif => ({ 
          ...notif, 
          read: true, 
          read_at: notif.read_at || new Date().toISOString() 
        }))
      );
      
      // Reset unread count
      setUnreadCount(0);
    } catch (err) {
      console.error('Failed to mark all notifications as read:', err);
    }
  };
  
  const handleNotificationClick = async (notification: Notification) => {
    if (!notification.read) {
      await handleMarkAsRead(notification.id);
    }
    
    // Handle navigation based on notification source
    if (notification.source === 'device' && notification.source_id) {
      // Navigate to device details
      window.location.href = `/devices?id=${notification.source_id}`;
    } else if (notification.source === 'package' && notification.source_id) {
      // Could navigate to package details if you have such a page
      // For now we'll just close the menu
      handleNotificationsClose();
    } else if (notification.source === 'model' && notification.source_id) {
      // Navigate to model details
      window.location.href = `/models?name=${notification.source_id}`;
    } else {
      // Default: just close the menu
      handleNotificationsClose();
    }
  };
  
  // Get notification badge color based on the priority of notifications
  const getNotificationBadgeColor = () => {
    // If there are error notifications, show red
    if (notifications.some(n => n.notification_type === 'error' && !n.read)) {
      return 'error';
    }
    
    // If there are warning notifications, show warning color
    if (notifications.some(n => n.notification_type === 'warning' && !n.read)) {
      return 'warning';
    }
    
    // Default is primary color for info/success
    return 'error';
  };
  
  return (
    <AppBar 
      position="fixed" 
      color="default" 
      elevation={0}
      sx={{ 
        zIndex: theme => theme.zIndex.drawer - 1, // Important: below the drawer
        borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
        bgcolor: sidebarOpen ? 'primary.dark' : 'white',
        color: sidebarOpen ? 'white' : 'primary.main',
        width: { sm: `calc(100% - ${sidebarOpen ? 240 : 60}px)` }, // Adjust width
        ml: { sm: `${sidebarOpen ? 240 : 60}px` }, // Adjust left margin
        transition: theme.transitions.create(['margin-left', 'width', 'background-color', 'color'], {
          easing: theme.transitions.easing.sharp,
          duration: theme.transitions.duration.enteringScreen,
        }),
        height: 64, // Fixed height to match sidebar header
      }}
    >
      <Toolbar sx={{ height: '100%' }}>
        {/* Only show menu icon if onMenuClick is provided */}
        {onMenuClick && (
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={onMenuClick}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
        )}
        
        {/* Breadcrumbs navigation */}
        <Box 
          sx={{ 
            flexGrow: 1, 
            '& .MuiBreadcrumbs-ol': {
              alignItems: 'center'
            },
            '& .MuiBreadcrumbs-li': {
              alignItems: 'center',
              display: 'flex'
            }
          }}
        >
          {generateBreadcrumbs()}
        </Box>
        
        <Box sx={{ display: 'flex' }}>
          <Tooltip title="Refresh">
            <IconButton 
              color="inherit" 
              onClick={handleRefresh} 
              size="large"
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Notifications">
            <IconButton
              color="inherit"
              onClick={handleNotificationsOpen}
              size="large"
            >
              <Badge badgeContent={unreadCount} color={getNotificationBadgeColor()}>
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>
        </Box>
        
        <Menu
          anchorEl={notificationsAnchor}
          open={Boolean(notificationsAnchor)}
          onClose={handleNotificationsClose}
          PaperProps={{
            elevation: 3,
            sx: { width: 360, maxHeight: 400, overflow: 'auto' }
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">
              Notifications {unreadCount > 0 && `(${unreadCount})`}
            </Typography>
            {unreadCount > 0 && (
              <Typography 
                variant="caption" 
                sx={{ cursor: 'pointer', color: 'primary.main' }} 
                onClick={handleMarkAllAsRead}
              >
                Mark all as read
              </Typography>
            )}
          </Box>
          <Divider />
          
          {isLoading && <LinearProgress />}
          
          {error && (
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="error">{error}</Typography>
            </Box>
          )}
          
          {!isLoading && !error && notifications.length === 0 && (
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">No notifications</Typography>
            </Box>
          )}
          
          {!isLoading && !error && notifications.length > 0 && (
            notifications.map((notification) => (
              <MenuItem 
                key={notification.id} 
                sx={{ 
                  py: 1, 
                  px: 2, 
                  alignItems: 'flex-start',
                  borderLeft: notification.read ? 'none' : '3px solid',
                  borderLeftColor: 
                    notification.notification_type === 'error' ? 'error.main' :
                    notification.notification_type === 'warning' ? 'warning.main' :
                    notification.notification_type === 'success' ? 'success.main' : 
                    'primary.main',
                  backgroundColor: notification.read ? 'inherit' : 'rgba(0, 0, 0, 0.04)'
                }}
                onClick={() => handleNotificationClick(notification)}
              >
                <Box sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography 
                      variant="body2" 
                      color={
                        notification.notification_type === 'error' ? 'error' :
                        notification.notification_type === 'warning' ? 'warning.main' :
                        notification.notification_type === 'success' ? 'success.main' : 
                        'primary'
                      }
                    >
                      {notification.notification_type.toUpperCase()}
                    </Typography>
                    <IconButton 
                      size="small" 
                      onClick={(e) => handleDismissNotification(notification.id, e)}
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </Box>
                  <Typography variant="body2">{notification.message}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatDate(notification.created_at)}
                  </Typography>
                </Box>
              </MenuItem>
            ))
          )}
        </Menu>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar;