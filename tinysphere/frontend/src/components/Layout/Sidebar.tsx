// src/components/Layout/Sidebar.tsx
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CodeIcon from '@mui/icons-material/Code';
import GitHubIcon from '@mui/icons-material/GitHub';
import HomeIcon from '@mui/icons-material/Home';
import MemoryIcon from '@mui/icons-material/Memory';
import MenuIcon from '@mui/icons-material/Menu';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import PsychologyIcon from '@mui/icons-material/Psychology';
import TimelineIcon from '@mui/icons-material/Timeline';
import {
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Typography,
  useTheme
} from '@mui/material';
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import logoImage from '../../assets/logo.png';

const DRAWER_WIDTH = 240;
const COLLAPSED_DRAWER_WIDTH = 60;

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  variant: 'permanent' | 'persistent' | 'temporary';
}

// Menu item type definition
interface MenuItem {
  text: string;
  icon: React.ReactNode;
  path: string;
  external?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onToggle, variant }) => {
  const location = useLocation();
  const theme = useTheme();
  
  const isActive = (path: string) => location.pathname === path;
  
  // Section 1: Main navigation
  const navigationItems: MenuItem[] = [
    { text: 'Home', icon: <HomeIcon />, path: '/', external: false },
    { text: 'Devices', icon: <MemoryIcon />, path: '/devices', external: false },
    { text: 'Models', icon: <PsychologyIcon />, path: '/models', external: false },
    { text: 'Drift', icon: <TimelineIcon />, path: '/drift', external: false },
    
  ];
  
  // Section 2: Applications
  const applicationItems: MenuItem[] = [
    { text: 'MLflow', icon: <ModelTrainingIcon />, path: '/mlflow', external: false },
    { text: 'MinIO', icon: <CloudUploadIcon />, path: '/minio', external: false },
  ];
  
  // Section 3: Documentation and links
  const resourceItems: MenuItem[] = [
    { text: 'Documentation', icon: <MenuBookIcon />, path: '/documentation', external: false },
    { text: 'API Docs', icon: <CodeIcon />, path: '/api-docs', external: false },
    { text: 'GitHub', icon: <GitHubIcon />, path: 'https://github.com/lukasgro63/tinysphere', external: true },
  ];

  // Logo section with improved styling and tooltip - for expanded sidebar
  const expandedLogo = (
    <Box sx={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center', 
      p: 2, 
      bgcolor: 'primary.dark',
      height: 64, // Match TopBar height exactly
    }}>
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'flex-start',
      }}>
        <img 
          src={logoImage} 
          alt="TinySphere Logo" 
          style={{ 
            height: 32, 
            width: 'auto', 
            marginRight: 12 
          }}
        />
        <Typography variant="h6" color="white" fontWeight="bold" noWrap>
          TinySphere
        </Typography>
      </Box>
      
      <Tooltip title="Collapse sidebar" placement="right">
        <IconButton onClick={onToggle} sx={{ color: 'white' }}>
          <ChevronLeftIcon />
        </IconButton>
      </Tooltip>
    </Box>
  );

  // Logo section for collapsed sidebar (only logo at top)
  const collapsedLogo = (
    <Box sx={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      p: 1, 
      bgcolor: 'primary.dark',
      height: 64, // Match TopBar height exactly
    }}>
      <img 
        src={logoImage} 
        alt="TinySphere Logo" 
        style={{ 
          height: 32, 
          width: 'auto'
        }}
      />
    </Box>
  );

  // Toggle button for collapsed sidebar in separate section
  const collapsedToggle = (
    <Box sx={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      p: 1, 
      bgcolor: 'primary.main',
      height: 50,  // Separate height for toggle button
      borderBottom: '1px solid rgba(255, 255, 255, 0.2)'
    }}>
      <Tooltip title="Expand sidebar" placement="right">
        <IconButton onClick={onToggle} sx={{ color: 'white' }}>
          <MenuIcon />
        </IconButton>
      </Tooltip>
    </Box>
  );

  // Style for active menu items with blue left border and background highlight
  const getListItemStyle = (active: boolean) => ({
    color: active ? 'primary.main' : 'primary.dark',
    borderLeft: `4px solid ${active ? theme.palette.primary.main : 'transparent'}`,
    backgroundColor: active ? 'rgba(11, 42, 90, 0.08)' : 'transparent',
    '&:hover': {
      backgroundColor: 'rgba(11, 42, 90, 0.05)',
    },
    transition: 'all 0.2s ease'
  });

  // Helper function for rendering menu items
  const renderMenuItems = (items: MenuItem[]) => (
    items.map((item) => {
      const active = isActive(item.path);
      return (
        <ListItemButton
          key={item.text}
          component={item.external ? 'a' : Link}
          to={!item.external ? item.path : undefined}
          href={item.external ? item.path : undefined}
          target={item.external ? "_blank" : undefined}
          rel={item.external ? "noopener noreferrer" : undefined}
          sx={getListItemStyle(active)}
        >
          <ListItemIcon sx={{ 
            color: active ? 'primary.main' : 'primary.dark', 
            minWidth: 36 
          }}>
            {item.icon}
          </ListItemIcon>
          <ListItemText 
            primary={item.text} 
            primaryTypographyProps={{ 
              fontWeight: active ? 'medium' : 'normal',
              color: active ? 'primary.main' : 'primary.dark'
            }}
          />
        </ListItemButton>
      );
    })
  );

  // Helper function for rendering collapsed menu items with WHITE active indicator
  const renderCollapsedMenuItems = (items: MenuItem[]) => (
    items.map((item) => {
      const active = isActive(item.path);
      return (
        <Tooltip title={item.text} placement="right" key={item.text}>
          <ListItemButton
            component={item.external ? 'a' : Link}
            to={!item.external ? item.path : undefined}
            href={item.external ? item.path : undefined}
            target={item.external ? "_blank" : undefined}
            rel={item.external ? "noopener noreferrer" : undefined}
            sx={{
              justifyContent: 'center',
              borderLeft: `4px solid ${active ? '#FFFFFF' : 'transparent'}`,
              backgroundColor: active ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
              },
              transition: 'all 0.2s ease'
            }}
          >
            <ListItemIcon sx={{ 
              color: active ? 'white' : 'rgba(255, 255, 255, 0.7)', 
              minWidth: 'auto',
            }}>
              {item.icon}
            </ListItemIcon>
          </ListItemButton>
        </Tooltip>
      );
    })
  );

  // Normal sidebar content
  const normalContent = (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%', 
      bgcolor: '#f0f2f5',
      color: 'text.primary',
      boxShadow: '1px 0 5px rgba(0,0,0,0.1)' 
    }}>
      {expandedLogo}
      
      {/* Section 1: Main navigation */}
      <List sx={{ pt: 0 }}>
        {renderMenuItems(navigationItems)}
      </List>
      
      {/* Corrected Divider */}
      <Box sx={{ px: 2, mt: 2, mb: 2 }}>
        <Divider sx={{ bgcolor: 'rgba(0, 0, 0, 0.1)' }} />
      </Box>
      
      {/* Section 2: Applications */}
      <List sx={{ pt: 0 }}>
        {renderMenuItems(applicationItems)}
      </List>
      
      {/* Corrected Divider */}
      <Box sx={{ px: 2, mt: 2, mb: 2 }}>
        <Divider sx={{ bgcolor: 'rgba(0, 0, 0, 0.1)' }} />
      </Box>
      
      {/* Section 3: Documentation and links */}
      <List>
        {renderMenuItems(resourceItems)}
      </List>
      
      <Box sx={{ flexGrow: 1 }} />
      
      <Box sx={{ p: 2, borderTop: '1px solid rgba(0, 0, 0, 0.1)' }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Version 0.1.0
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5 }}>
            © 2025 Powered by <Link to="https://github.com/lukasgro63/tinylcm" target="_blank" rel="noopener noreferrer" style={{ color: theme.palette.primary.main }}>TinyLCM</Link>
          </Typography>
        </Box>
      </Box>
    </Box>

  );

  // Collapsed sidebar content
  const collapsedContent = (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%', 
      bgcolor: 'primary.main', 
      color: 'white' 
    }}>
      {/* Logo at the very top */}
      {collapsedLogo}
      
      {/* Toggle button in separate section - BEFORE navigation items */}
      {collapsedToggle}
      
      {/* Section 1: Main navigation */}
      <List sx={{ pt: 0 }}>
        {renderCollapsedMenuItems(navigationItems)}
      </List>
      
      <Divider sx={{ my: 2, bgcolor: 'rgba(255, 255, 255, 0.2)' }} />
      
      {/* Section 2: Applications */}
      <List sx={{ pt: 0 }}>
        {renderCollapsedMenuItems(applicationItems)}
      </List>
      
      <Divider sx={{ my: 2, bgcolor: 'rgba(255, 255, 255, 0.2)' }} />
      
      {/* Section 3: Documentation and links */}
      <List>
        {renderCollapsedMenuItems(resourceItems)}
      </List>
      
      <Box sx={{ flexGrow: 1 }} />
    </Box>
  );

  return (
    <Drawer
      variant={variant}
      open={variant === 'temporary' ? isOpen : true}
      onClose={variant === 'temporary' ? onToggle : undefined}
      sx={{
        width: isOpen ? DRAWER_WIDTH : COLLAPSED_DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: isOpen ? DRAWER_WIDTH : COLLAPSED_DRAWER_WIDTH,
          boxSizing: 'border-box',
          bgcolor: isOpen ? '#f0f2f5' : 'primary.main',
          overflowX: 'hidden',
          transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
          borderRight: isOpen ? '1px solid rgba(0, 0, 0, 0.12)' : 'none',
        },
      }}
    >
      {isOpen ? normalContent : collapsedContent}
    </Drawer>
  );
};

export default Sidebar;