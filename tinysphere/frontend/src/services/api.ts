// src/services/api.ts
import axios from 'axios';
import {
  Device,
  DeviceMetrics,
  DeviceSummary,
  DeviceTrend,
  DriftEvent,
  DriftImage,
  DriftImagesResponse,
  DriftSample,
  DriftStatistics,
  DriftValidation,
  Model,
  ModelPerformanceData,
  ModelSummary,
  Notification,
  NotificationCount,
  NotificationResponse,
  Package,
  PackageActivity,
  PackageTimelineData,
  PlatformDistribution,
  SystemStatus,
  TopDevice,
  PredictionImage,
  PredictionImagesResponse,
  ImageUrlResponse
} from '../types/api';

const API_BASE_URL = '/api';

// Devices
export const getDevices = async () => {
  const response = await axios.get<Device[]>(`${API_BASE_URL}/devices/`);
  return response.data;
};

export const getDevice = async (deviceId: string) => {
  const response = await axios.get<Device>(`${API_BASE_URL}/devices/${deviceId}`);
  return response.data;
};

// Packages
export const getPackages = async () => {
  const response = await axios.get<Package[]>(`${API_BASE_URL}/packages/`);
  return response.data;
};

export const getPackagesByDevice = async (deviceId: string) => {
  const response = await axios.get<Package[]>(`${API_BASE_URL}/packages/device/${deviceId}`);
  return response.data;
};

// Models
export const getModels = async () => {
  const response = await axios.get<string[]>(`${API_BASE_URL}/models/`);
  return response.data;
};

export const getModelVersions = async (modelName: string) => {
  const response = await axios.get<Model[]>(`${API_BASE_URL}/models/${modelName}/versions`);
  return response.data;
};

export const getAllModelVersions = async () => {
  // Hole zuerst alle Modellnamen
  const models = await getModels();
  
  // Dann hole für jedes Modell die Versionen
  const allVersionsPromises = models.map(model => getModelVersions(model));
  const allVersionsResults = await Promise.all(allVersionsPromises);
  
  // Kombiniere alle Versionen in eine flache Liste
  return allVersionsResults.flat();
};

// Dashboard
export const getSystemStatus = async () => {
  const response = await axios.get<SystemStatus>(`${API_BASE_URL}/dashboard/status`);
  return response.data;
};

export const getDevicesSummary = async () => {
  const response = await axios.get<DeviceSummary[]>(`${API_BASE_URL}/dashboard/devices/summary`);
  return response.data;
};

export const getModelsSummary = async () => {
  const response = await axios.get<ModelSummary[]>(`${API_BASE_URL}/dashboard/models/summary`);
  return response.data;
};

export const getPackageActivity = async (limit: number = 10) => {
  const response = await axios.get<PackageActivity[]>(`${API_BASE_URL}/dashboard/packages/activity?limit=${limit}`);
  return response.data;
};

export const getPackageTimeline = async (period: string = 'week'): Promise<PackageTimelineData[]> => {
  const response = await axios.get<PackageTimelineData[]>(`${API_BASE_URL}/dashboard/packages/timeline?period=${period}`);
  return response.data;
};

export const getModelsPerformance = async (
  metric: string = 'accuracy',
  days?: number,
  limit?: number,
  model_name?: string,
  tags?: string,
  include_operational_metrics: boolean = true
): Promise<ModelPerformanceData[]> => {
  let url = `${API_BASE_URL}/dashboard/models/performance?metric=${metric}`;

  // Add optional filters if provided
  if (days !== undefined) url += `&days=${days}`;
  if (limit !== undefined) url += `&limit=${limit}`;
  if (model_name) url += `&model_name=${encodeURIComponent(model_name)}`;
  if (tags) url += `&tags=${encodeURIComponent(tags)}`;
  if (include_operational_metrics !== undefined) url += `&include_operational_metrics=${include_operational_metrics}`;

  const response = await axios.get<ModelPerformanceData[]>(url);
  return response.data;
};

export const getModelMetrics = async (modelName: string, metric: string = 'accuracy'): Promise<any[]> => {
  const response = await axios.get<any[]>(`${API_BASE_URL}/models/${modelName}/metrics?metric=${metric}`);
  return response.data;
};

export const compareModelVersions = async (modelName: string, versions: number[]): Promise<any[]> => {
  const response = await axios.post<any[]>(`${API_BASE_URL}/models/compare`, {
    model_name: modelName,
    versions: versions
  });
  return response.data;
};

// Notifications
export const getNotifications = async (skip: number = 0, limit: number = 50, unreadOnly: boolean = false): Promise<NotificationResponse> => {
  const response = await axios.get<NotificationResponse>(
    `${API_BASE_URL}/notifications/?skip=${skip}&limit=${limit}&unread_only=${unreadOnly}`
  );
  return response.data;
};

export const getNotificationCounts = async (): Promise<NotificationCount> => {
  const response = await axios.get<NotificationCount>(`${API_BASE_URL}/notifications/count`);
  return response.data;
};

export const markNotificationAsRead = async (notificationId: number): Promise<Notification> => {
  const response = await axios.patch<Notification>(
    `${API_BASE_URL}/notifications/${notificationId}`,
    { read: true }
  );
  return response.data;
};

export const markAllNotificationsAsRead = async (): Promise<{ success: boolean; count: number; action: string }> => {
  const response = await axios.post(`${API_BASE_URL}/notifications/actions`, {
    action: "mark_all_read",
    notification_ids: []
  });
  return response.data;
};

export const dismissNotification = async (notificationId: number): Promise<{ success: boolean; message: string }> => {
  const response = await axios.delete(`${API_BASE_URL}/notifications/${notificationId}`);
  return response.data;
};

// Get platform distribution data for charts
export const getDevicePlatforms = async (): Promise<PlatformDistribution[]> => {
  const response = await axios.get<PlatformDistribution[]>(`${API_BASE_URL}/devices/platforms`);
  return response.data;
};

// Get device metrics for the dashboard cards
export const getDeviceMetrics = async (deviceId?: string): Promise<DeviceMetrics> => {
  const url = `${API_BASE_URL}/devices/metrics${deviceId ? `?device_id=${deviceId}` : ''}`;
  const response = await axios.get<DeviceMetrics>(url);
  return response.data;
};

// Get device connectivity trends for charts
export const getDeviceConnectivityTrends = async (days: number = 7): Promise<DeviceTrend[]> => {
  const response = await axios.get<DeviceTrend[]>(`${API_BASE_URL}/devices/trends?days=${days}`);
  return response.data;
};

// Get top devices by package count
export const getTopDevices = async (limit: number = 5): Promise<TopDevice[]> => {
  const response = await axios.get<TopDevice[]>(`${API_BASE_URL}/devices/top?limit=${limit}`);
  return response.data;
};

// Prediction Images API methods
export const getPredictionImageDevices = async (): Promise<string[]> => {
  const response = await axios.get<string[]>(`${API_BASE_URL}/prediction-images/devices`);
  return response.data;
};

export const getPredictionTypes = async (deviceId: string): Promise<string[]> => {
  const response = await axios.get<string[]>(
    `${API_BASE_URL}/prediction-images/devices/${deviceId}/types`
  );
  return response.data;
};

export const getPredictionDates = async (deviceId: string, predictionType: string): Promise<string[]> => {
  const response = await axios.get<string[]>(
    `${API_BASE_URL}/prediction-images/devices/${deviceId}/types/${predictionType}/dates`
  );
  return response.data;
};

export const getPredictionImages = async (
  deviceId?: string,
  predictionType?: string,
  date?: string,
  limit: number = 100,
  offset: number = 0,
  sortOrder?: 'asc' | 'desc'
): Promise<PredictionImagesResponse> => {
  let url = `${API_BASE_URL}/prediction-images/list?limit=${limit}&offset=${offset}`;

  if (deviceId) url += `&device_id=${deviceId}`;
  if (predictionType) url += `&prediction_type=${predictionType}`;
  if (date) url += `&date=${date}`;
  if (sortOrder) url += `&sort_order=${sortOrder}`;

  const response = await axios.get<PredictionImagesResponse>(url);
  
  // Client-seitige Sortierung, falls das Backend die Sortierung nicht unterstützt
  if (sortOrder && response.data.images.length > 0) {
    const sortedImages = [...response.data.images].sort((a, b) => {
      const dateA = new Date(a.last_modified).getTime();
      const dateB = new Date(b.last_modified).getTime();
      
      return sortOrder === 'desc' 
        ? dateB - dateA  // Newest first (desc)
        : dateA - dateB; // Oldest first (asc)
    });
    
    return {
      ...response.data,
      images: sortedImages
    };
  }
  
  return response.data;
};

export const getPredictionImageUrl = async (imageKey: string): Promise<ImageUrlResponse> => {
  const response = await axios.get<ImageUrlResponse>(
    `${API_BASE_URL}/prediction-images/url/${encodeURIComponent(imageKey)}`
  );
  return response.data;
};

// Drift Management
export const getDriftEvents = async (params?: {
  skip?: number;
  limit?: number;
  device_id?: string;
  status?: string;
  drift_type?: string;
  start_date?: string;
  end_date?: string;
  sort_order?: 'asc' | 'desc'; // Neue Parameter für Sortierung
}): Promise<DriftEvent[]> => {
  const queryParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        // Special handling for drift_type to ensure it's lowercase
        if (key === 'drift_type' && typeof value === 'string') {
          queryParams.append(key, value.toLowerCase());
        } else {
          queryParams.append(key, value.toString());
        }
      }
    });
  }
  
  const response = await axios.get<DriftEvent[]>(
    `${API_BASE_URL}/drift/events?${queryParams.toString()}`
  );
  
  // Client-seitige Sortierung als Fallback, falls Backend-API keine Sortierung unterstützt
  if (params?.sort_order) {
    const sortedData = [...response.data].sort((a, b) => {
      const dateA = new Date(a.timestamp).getTime();
      const dateB = new Date(b.timestamp).getTime();
      
      return params.sort_order === 'desc' 
        ? dateB - dateA  // Newest first (desc)
        : dateA - dateB; // Oldest first (asc)
    });
    return sortedData;
  }
  
  return response.data;
};

export const getDriftEvent = async (eventId: string): Promise<DriftEvent> => {
  const response = await axios.get<DriftEvent>(`${API_BASE_URL}/drift/events/${eventId}`);
  return response.data;
};

export const getDriftSamples = async (eventId: string): Promise<DriftSample[]> => {
  const response = await axios.get<DriftSample[]>(
    `${API_BASE_URL}/drift/events/${eventId}/samples`
  );
  return response.data;
};

export const getDriftSample = async (sampleId: string): Promise<DriftSample> => {
  const response = await axios.get<DriftSample>(
    `${API_BASE_URL}/drift/samples/${sampleId}`
  );
  return response.data;
};

export const validateDriftSample = async (
  sampleId: string, 
  validation: {
    is_valid_drift: boolean;
    true_label?: string;
    validated_by?: string;
    notes?: string;
  }
): Promise<DriftValidation> => {
  const response = await axios.post<DriftValidation>(
    `${API_BASE_URL}/drift/samples/${sampleId}/validate`,
    validation
  );
  return response.data;
};

export const updateDriftEventStatus = async (
  eventId: string,
  status: string,
  resolution_notes?: string
): Promise<DriftEvent> => {
  const response = await axios.patch<DriftEvent>(
    `${API_BASE_URL}/drift/events/${eventId}/status`,
    { status, resolution_notes }
  );
  return response.data;
};

export const getDriftStatistics = async (
  device_id?: string,
  days: number = 30,
  drift_type?: string
): Promise<DriftStatistics> => {
  const params = new URLSearchParams();
  if (device_id) {
    params.append('device_id', device_id);
  }
  params.append('days', days.toString());
  
  // Add drift_type parameter if provided, ensuring it's lowercase
  if (drift_type) {
    params.append('drift_type', drift_type.toLowerCase());
  }

  const response = await axios.get<DriftStatistics>(
    `${API_BASE_URL}/drift/statistics?${params.toString()}`
  );
  return response.data;
};

// Drift Images API methods
export const getDriftImageDevices = async (): Promise<string[]> => {
  const response = await axios.get<string[]>(`${API_BASE_URL}/drift-images/devices`);
  return response.data;
};

export const getDriftTypes = async (deviceId: string): Promise<string[]> => {
  const response = await axios.get<string[]>(
    `${API_BASE_URL}/drift-images/devices/${deviceId}/types`
  );
  return response.data;
};

export const getDriftDates = async (deviceId: string, driftType: string): Promise<string[]> => {
  const response = await axios.get<string[]>(
    `${API_BASE_URL}/drift-images/devices/${deviceId}/types/${driftType}/dates`
  );
  return response.data;
};

export const getDriftImages = async (
  deviceId?: string,
  driftType?: string,
  date?: string,
  limit: number = 100,
  offset: number = 0,
  sortOrder?: 'asc' | 'desc'
): Promise<DriftImagesResponse> => {
  let url = `${API_BASE_URL}/drift-images/list?limit=${limit}&offset=${offset}`;

  if (deviceId) url += `&device_id=${deviceId}`;
  if (driftType) url += `&drift_type=${driftType}`;
  if (date) url += `&date=${date}`;
  if (sortOrder) url += `&sort_order=${sortOrder}`;

  const response = await axios.get<DriftImagesResponse>(url);
  
  // Client-seitige Sortierung, falls das Backend die Sortierung nicht unterstützt
  if (sortOrder && response.data.images.length > 0) {
    const sortedImages = [...response.data.images].sort((a, b) => {
      const dateA = new Date(a.last_modified).getTime();
      const dateB = new Date(b.last_modified).getTime();
      
      return sortOrder === 'desc' 
        ? dateB - dateA  // Newest first (desc)
        : dateA - dateB; // Oldest first (asc)
    });
    
    return {
      ...response.data,
      images: sortedImages
    };
  }
  
  return response.data;
};

export const getDriftImageUrl = async (imageKey: string): Promise<ImageUrlResponse> => {
  const response = await axios.get<ImageUrlResponse>(
    `${API_BASE_URL}/drift-images/url/${encodeURIComponent(imageKey)}`
  );
  return response.data;
};
