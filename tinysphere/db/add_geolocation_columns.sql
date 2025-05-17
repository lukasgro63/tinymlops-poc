-- SQL to manually add geolocation columns to the devices table

-- Check if columns already exist before adding them
DO $$
DECLARE
    latitude_exists BOOLEAN;
    longitude_exists BOOLEAN;
    geo_accuracy_exists BOOLEAN;
BEGIN
    -- Check if columns exist
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'devices' AND column_name = 'latitude'
    ) INTO latitude_exists;
    
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'devices' AND column_name = 'longitude'
    ) INTO longitude_exists;
    
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'devices' AND column_name = 'geo_accuracy'
    ) INTO geo_accuracy_exists;
    
    -- Add latitude column if it doesn't exist
    IF NOT latitude_exists THEN
        ALTER TABLE devices ADD COLUMN latitude FLOAT;
        RAISE NOTICE 'Added latitude column';
    ELSE
        RAISE NOTICE 'latitude column already exists';
    END IF;
    
    -- Add longitude column if it doesn't exist
    IF NOT longitude_exists THEN
        ALTER TABLE devices ADD COLUMN longitude FLOAT;
        RAISE NOTICE 'Added longitude column';
    ELSE
        RAISE NOTICE 'longitude column already exists';
    END IF;
    
    -- Add geo_accuracy column if it doesn't exist
    IF NOT geo_accuracy_exists THEN
        ALTER TABLE devices ADD COLUMN geo_accuracy FLOAT;
        RAISE NOTICE 'Added geo_accuracy column';
    ELSE
        RAISE NOTICE 'geo_accuracy column already exists';
    END IF;
END $$;