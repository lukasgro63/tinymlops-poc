// src/types/api.ts
export interface Device {
  id: number;
  device_id: string;
  hostname?: string;
  ip_address?: string;
  platform?: string;
  platform_version?: string;  // Added for OS version (e.g., "bookworm")
  device_model?: string;      // Added for device model (e.g., "Pi Zero")
  python_version?: string;
  tinylcm_version?: string;
  registration_time: string;
  last_sync_time?: string;
  is_active: boolean;
  device_info?: Record<string, any>;
  latitude?: number;
  longitude?: number;
  location_name?: string;
  last_location_update?: string;
}

// Drift management types
export type DriftType = 'confidence' | 'distribution' | 'feature' | 'outlier' | 'custom' | 'knn_distance' | 'unknown';
export type DriftStatus = 'pending' | 'validated' | 'rejected' | 'resolved' | 'ignored';

export interface DriftEvent {
  event_id: string;
  device_id: string;
  drift_type: DriftType;
  drift_score?: number;
  detector_name?: string;
  model_id?: string;
  description?: string;
  status: DriftStatus;
  timestamp: string;
  received_at: string;
  resolved_at?: string;
  resolution_notes?: string;
  sample_count: number;
  validation_count: number;
  metadata?: {
    original_drift_type?: string;
    drift_type_display?: string;
    [key: string]: any;
  };
}

export interface DriftSample {
  sample_id: string;
  drift_event_id: string;
  prediction?: string;
  confidence?: number;
  drift_score?: number;
  feature_path?: string;
  raw_data_path?: string;
  timestamp?: string;
  true_label?: string;
  status: 'pending' | 'validated';
  metadata?: Record<string, any>;
}

export interface DriftValidation {
  validation_id: string;
  drift_event_id: string;
  drift_sample_id?: string;
  is_valid_drift: boolean;
  true_label?: string;
  validated_by?: string;
  validated_at: string;
  is_acknowledged: boolean;
  acknowledged_at?: string;
  notes?: string;
}

export interface DriftStatistics {
  total_events: number;
  total_open: number;
  total_resolved: number;
  by_type: Record<string, number>;
  by_status: Record<DriftStatus, number>;
  by_day: Array<{
    date: string;
    count: number;
  }>;
  recent_events: DriftEvent[];
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
  platform_version?: string;  // Added for OS version (e.g., "bookworm")
  device_model?: string;      // Added for device model (e.g., "Pi Zero")
  is_active: boolean;
  last_sync_time?: string;
  package_count: number;
  latest_package?: string;
  latitude?: number;
  longitude?: number;
  location_name?: string;
  last_location_update?: string;
  mlflow_metrics?: {
    inference_time?: {
      avg: number;
      min: number;
      max: number;
      count: number;
    };
    cpu_usage?: {
      avg: number;
      min: number;
      max: number;
      count: number;
    };
    memory_usage?: {
      avg: number;
      min: number;
      max: number;
      count: number;
    };
    drift_score?: number[];
    available_metrics?: string[];
  };
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
  drift_events: number;
  prediction_images: number;
}

export interface ModelPerformanceData {
  model_name: string;
  version: number;
  stage: string;
  metric_name: string;
  value: number | null;  // Allow null for missing metrics
  timestamp: number;
  run_id?: string;
}

export interface DevicePerformanceData {
  device_id: string;
  metric_name: string;
  value: number | null;  // Allow null for missing metrics
  timestamp: number;
  version?: number;      // For consistency with model data format
  run_id?: string;
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

// Prediction Images Types
export interface PredictionImage {
  key: string;
  device_id: string;
  prediction_type: string;
  date: string;
  filename: string;
  size: number;
  last_modified: string;
  url: string;
}

export interface PredictionImagesResponse {
  total: number;
  images: PredictionImage[];
}

// Drift Images Types
export interface DriftImage {
  key: string;
  device_id: string;
  drift_type: string;
  date: string;
  filename: string;
  size: number;
  last_modified: string;
  url: string;
  event_id?: string;
  prediction_type?: string; // Added for type compatibility with PredictionImage
}

export interface DriftImagesResponse {
  total: number;
  images: DriftImage[];
}

export interface ImageUrlResponse {
  url: string;
}

// Operational Logs Types
export interface OperationalLog {
  key: string;
  device_id: string;
  session_id: string;
  filename: string;
  size: number;
  last_modified: string;
  url: string;
  date?: string; // Date in YYYYMMDD format
  log_type?: string; // Type of the operational log (e.g., "system", "metrics", "inference")
  is_consolidated?: boolean; // Whether this log is a consolidated log file
}

export interface OperationalLogResponse {
  total: number;
  logs: OperationalLog[];
}

export interface LogDeletionResponse {
  status: string;
  message: string;
  deleted_count: number;
  error_count?: number;
  errors?: Array<{Key: string; Code: string; Message: string}>;
}

// Device location types
export interface DeviceLocation {
  device_id: string;
  latitude: number;
  longitude: number;
  location_name?: string;
  last_update: string;
}

export interface DeviceLocationsResponse {
  total: number;
  locations: DeviceLocation[];
}
