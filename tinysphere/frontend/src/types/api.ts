// src/types/api.ts
export interface Device {
  id: number;
  device_id: string;
  hostname?: string;
  ip_address?: string;
  platform?: string;
  python_version?: string;
  tinylcm_version?: string;
  registration_time: string;
  last_sync_time?: string;
  is_active: boolean;
  device_info?: Record<string, any>;
}

export interface Package {
  id: number;
  package_id: string;
  device_id: string;
  package_type: string;
  filename: string;
  file_path: string;
  file_hash?: string;
  file_size?: number;
  uploaded_at: string;
  processed_at?: string;
  is_processed: boolean;
  processing_status: string;
  processing_error?: string;
  package_metadata?: Record<string, any>;
}

export interface Model {
  name: string;
  version: number;
  stage: string;
  source?: string;
  run_id?: string;
}

export interface SystemStatus {
  status: string;
  statistics: {
    total_devices: number;
    active_devices: number;
    total_packages: number;
    processed_packages: number;
    pending_packages: number;
    package_types: Record<string, number>;
  };
}

export interface DeviceSummary {
  device_id: string;
  hostname?: string;
  platform?: string;
  is_active: boolean;
  last_sync_time?: string;
  package_count: number;
  latest_package?: string;
}

export interface ModelSummary {
  name: string;
  total_versions: number;
  latest_version: number;
  has_production: boolean;
  has_staging: boolean;
  production_version?: number;
  staging_version?: number;
  device_id?: string;
}

export interface PackageActivity {
  package_id: string;
  device_id: string;
  device_name: string;
  package_type: string;
  uploaded_at: string;
  processed_at?: string;
  is_processed: boolean;
  processing_status: string;
}

export interface PackageTimelineData {
  date: string;
  models: number;
  metrics: number;
  data_logs: number;
}

export interface ModelPerformanceData {
  model_name: string;
  version: number;
  stage: string;
  metric_name: string;
  value: number;
  timestamp: number;
}

export interface ModelVersionMetrics {
  version: number;
  stage: string;
  run_id: string;
  created_at: number;
  metrics: Record<string, number>;
}

export interface MetricRow {
  metric: string;
  values: Record<number, number | null>;
  diff?: number;
}

// Notification types
export type NotificationType = 'info' | 'warning' | 'error' | 'success';

export interface Notification {
  id: number;
  message: string;
  notification_type: NotificationType;
  source?: string;
  source_id?: string;
  details?: Record<string, any>;
  created_at: string;
  read: boolean;
  read_at?: string;
}

export interface NotificationResponse {
  items: Notification[];
  total: number;
  unread: number;
}

export interface NotificationCount {
  total: number;
  unread: number;
}

// Platform Distribution
export interface PlatformDistribution {
  name: string;
  value: number;
}

export interface DeviceMetrics {
  inference_time: {
    avg: number;
    min: number;
    max: number;
  };
  cpu_usage: {
    avg: number;
    min: number;
    max: number;
  };
  memory_usage: {
    avg: number;
    min: number;
    max: number;
  };
}

export interface DeviceTrend {
  date: string;
  active: number;
  total: number;
}

export interface TopDevice {
  device_id: string;
  name: string;
  packages: number;
}
