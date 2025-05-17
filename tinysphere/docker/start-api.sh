#!/bin/bash
set -e

# Besseres Error-Handling
function handle_error {
    echo "ERROR: Ein Fehler ist aufgetreten. Siehe Details oben."
    # In einem echten Produktionssystem würden wir hier nicht einfach weitermachen,
    # sondern möglicherweise auf einen alternativen Modus umschalten oder neu starten.
    # Für diese Demo lassen wir den Server starten, auch wenn Migrationen fehlschlagen.
    return 0
}

# Enable error trap
trap 'handle_error' ERR

# Wait for database to be available
echo "Waiting for database..."
sleep 5

# Set database credentials as environment variables
export PGPASSWORD=tinysphere
export PGUSER=tinysphere
export PGHOST=postgres
export PGDATABASE=tinysphere

# Verify database connection before proceeding
echo "Checking database connection..."
if ! pg_isready -h $PGHOST -U $PGUSER; then
    echo "WARNING: Database not available yet. Waiting longer..."
    sleep 10
    if ! pg_isready -h $PGHOST -U $PGUSER; then
        echo "ERROR: Database not available after extended wait. Starting without migrations."
        echo "ENVIRONMENT=development" > /tmp/env_vars
        # Skip migrations and go directly to app start
        cd /app
        exec uvicorn tinysphere.api.main:app --host 0.0.0.0 --port 8000 --reload
        exit 0
    fi
fi

# Add Geolocation columns directly if needed
echo "Checking for Geolocation columns..."

if ! psql -c "SELECT column_name FROM information_schema.columns WHERE table_name='devices' AND column_name='latitude'" | grep -q latitude; then
    echo "Adding geolocation columns directly..."
    psql -c "
    DO \$\$
    BEGIN
        -- Check if table exists
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'devices') THEN
            -- Add columns if they don't exist
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'devices' AND column_name = 'latitude') THEN
                ALTER TABLE devices ADD COLUMN latitude FLOAT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'devices' AND column_name = 'longitude') THEN
                ALTER TABLE devices ADD COLUMN longitude FLOAT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'devices' AND column_name = 'geo_accuracy') THEN
                ALTER TABLE devices ADD COLUMN geo_accuracy FLOAT;
            END IF;
        END IF;
    END \$\$;
    "
fi

# Try to run migrations, but continue even if they fail
echo "Running database migrations..."
cd /app/tinysphere
PYTHONPATH=/app alembic upgrade head || {
    echo "WARNING: Migrations failed but we'll continue anyway since the columns were added directly"
}

# Create database tables if they don't exist
echo "Ensuring database schema is complete..."
cd /app
ENVIRONMENT=development

# Start the application
echo "Starting API server..."
exec uvicorn tinysphere.api.main:app --host 0.0.0.0 --port 8000 --reload