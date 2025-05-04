// src/components/common/ModelComparisonTable.tsx
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';
import React from 'react';
import { MetricRow } from '../../types/api';

interface ModelComparisonTableProps {
  modelName: string;
  versions: number[];
  comparisonData: MetricRow[];
}

const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({ 
  modelName, 
  versions, 
  comparisonData 
}) => {
  if (versions.length === 0) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Select two versions to compare
        </Typography>
      </Box>
    );
  }
  
  if (comparisonData.length === 0) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          No comparison data available for the selected versions
        </Typography>
      </Box>
    );
  }

  // Format metric values
  const formatMetricValue = (value: number | null, metric: string) => {
    if (value === null) return 'N/A';
    
    if (metric.toLowerCase().includes('loss')) {
      return value.toFixed(4);
    } else if (value >= 0 && value <= 1) {
      return `${(value * 100).toFixed(2)}%`;
    } else {
      return value.toFixed(4);
    }
  };

  return (
    <Box>
      <Typography variant="subtitle1" gutterBottom>
        Comparing versions: {versions.join(' vs ')}
      </Typography>
      
      <TableContainer component={Paper} sx={{ maxHeight: 350, overflow: 'auto' }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Metric</TableCell>
              {versions.map(version => (
                <TableCell key={version} align="center">Version {version}</TableCell>
              ))}
              {versions.length === 2 && (
                <TableCell align="center">Difference</TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {comparisonData.map((row) => (
              <TableRow key={row.metric}>
                <TableCell component="th" scope="row">
                  {row.metric.charAt(0).toUpperCase() + row.metric.slice(1)}
                </TableCell>
                {versions.map(version => (
                  <TableCell key={version} align="center">
                    {formatMetricValue(row.values[version], row.metric)}
                  </TableCell>
                ))}
                {versions.length === 2 && row.diff !== undefined && (
                  <TableCell 
                    align="center"
                    sx={{ 
                      color: row.metric.toLowerCase().includes('loss') 
                        ? (row.diff < 0 ? 'success.main' : row.diff > 0 ? 'error.main' : 'inherit')
                        : (row.diff > 0 ? 'success.main' : row.diff < 0 ? 'error.main' : 'inherit')
                    }}
                  >
                    {row.diff > 0 ? '+' : ''}{formatMetricValue(row.diff, row.metric)}
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ModelComparisonTable;