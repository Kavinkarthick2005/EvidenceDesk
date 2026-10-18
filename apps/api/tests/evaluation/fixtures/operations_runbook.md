# EvidenceDesk Operations and Maintenance Runbook

## 1. Network Topology and PostgreSQL Port
The backend service connects to PostgreSQL on port 5432 over a bridge network in Docker Compose. The default database name is evidencedesk and the default database user is evidencedesk. Health checks ping port 5432 with pg_isready every 10 seconds.

## 2. Log Management and Rotation Limits
Application log files rotate automatically when they reach 100 MB in disk storage. Standard rotation retains up to 5 archived log files before purging oldest entries. The log level is configurable via APP_LOG_LEVEL with supported values DEBUG, INFO, WARNING, and ERROR.

## 3. Backup Schedules and Retention
Automated pg_dump database backups execute nightly at 02:00 UTC and save compressed SQL dumps into /var/backups/evidencedesk. Backup archives older than 30 days are purged by a daily cron script.

## 4. Health Check Endpoints
The FastAPI service exposes a health check endpoint at GET /health on port 8000. It returns an HTTP 200 response with JSON payload {"status": "healthy"} when the process is operational.

## 5. Host Memory Requirements
Docker Compose environments require a minimum of 4 GB of host RAM to run inference and PostgreSQL concurrently. Systems with less than 4 GB of memory may experience out-of-memory errors during PyTorch tensor allocation.

## 6. Container Restart Policies
All Docker containers in compose.yaml are configured with restart policy unless-stopped. If the database crashes unexpectedly, Docker automatically restarts the container within 5 seconds.
