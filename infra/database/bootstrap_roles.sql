-- Initial bootstrap for a NEW stock_predictor database, executed as schema_migrator.
-- These roles already exist in the provisioned project. Do not rerun blindly.
-- Supply runtime_password and writer_password as securely bound psql variables;
-- these are execution parameters, not additional application .env variables.
-- Actual passwords and connection URLs must never be committed here.

CREATE ROLE app_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS PASSWORD :'runtime_password';
CREATE ROLE pipeline_writer LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS PASSWORD :'writer_password';

GRANT CONNECT ON DATABASE stock_predictor TO app_runtime, pipeline_writer;
GRANT USAGE ON SCHEMA public TO app_runtime, pipeline_writer;

-- Grant permissions on specific application/pipeline tables in later schema
-- migrations. Do not grant either role neon_superuser or blanket admin rights.

