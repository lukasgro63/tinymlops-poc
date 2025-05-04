// src/components/Layout/MainLayout.tsx
import { Box, CssBaseline, Toolbar, useMediaQuery, useTheme } from '@mui/material';
import React, { useState } from 'react';
import Sidebar from './Sidebar';
import TopBar from './TopBar';

interface MainLayoutProps {
  title: string;
  children: React.ReactNode;
  isEmbedded?: boolean;
}

const MainLayout: React.FC<MainLayoutProps> = ({ title, children, isEmbedded = false }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Determine if TopBar should be shown
  const showTopBar = !isEmbedded || (isEmbedded && sidebarOpen);

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <CssBaseline />
      
      {/* Sidebar - always first in layout */}
      {isMobile ? (
        <Sidebar
          isOpen={sidebarOpen}
          onToggle={handleSidebarToggle}
          variant="temporary"
        />
      ) : (
        <Sidebar
          isOpen={sidebarOpen}
          onToggle={handleSidebarToggle}
          variant="permanent"
        />
      )}
      
      {/* Box for main content container */}
      <Box 
        sx={{ 
          flexGrow: 1, 
          width: { sm: `calc(100% - ${sidebarOpen ? 240 : 60}px)` },
          transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          backgroundColor: 'background.default',
        }}
      >
        {/* TopBar - only for main content */}
        {showTopBar && (
          <TopBar 
            title={title}
            onMenuClick={isMobile ? handleSidebarToggle : undefined}
            sidebarOpen={sidebarOpen}
          />
        )}
        
        {/* The actual main content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: isEmbedded ? 0 : 3,
            mt: showTopBar ? 8 : 0, // Space for the TopBar
            bgcolor: 'background.default',
          }}
        >
          {isMobile && <Toolbar />} {/* Space for the app bar on mobile */}
          
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default MainLayout;