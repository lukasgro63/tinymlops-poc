// src/services/api.ts
import axios from 'axios';
import {
  Device,
  DeviceMetrics,
  DeviceSummary,
  DeviceTrend,
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
  TopDevice
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

export const getModelsPerformance = async (metric: string = 'accuracy'): Promise<ModelPerformanceData[]> => {
  const response = await axios.get<ModelPerformanceData[]>(`${API_BASE_URL}/dashboard/models/performance?metric=${metric}`);
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
