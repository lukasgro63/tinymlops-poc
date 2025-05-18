-- Add platform_version and device_model columns to devices table
ALTER TABLE devices ADD COLUMN IF NOT EXISTS platform_version VARCHAR(255);
ALTER TABLE devices ADD COLUMN IF NOT EXISTS device_model VARCHAR(255);

-- Comment on columns for documentation
COMMENT ON COLUMN devices.platform_version IS 'OS version (e.g., bookworm, bullseye)';
COMMENT ON COLUMN devices.device_model IS 'Device model (e.g., Pi Zero, Pi 4, etc.)';