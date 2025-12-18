# Production Security Checklist

Complete security hardening checklist for deploying VideoAnnotator in production environments.

## Pre-Deployment Checklist

### 🔐 Authentication & Authorization

- [ ] **API Keys Generated**: Create production-specific API keys
  ```bash
  uv run python -m scripts.manage_tokens create \
    --user-id prod_admin \
    --username "Production Admin" \
    --email admin@yourdomain.com \
    --scopes read,write,admin \
    --expires-in-days 90
  ```

- [ ] **Key Expiration Set**: Production keys expire within 90 days
  ```bash
  # Verify expiration
  uv run python -m scripts.manage_tokens list
  ```

- [ ] **Test Keys Removed**: Delete all development/test keys
  ```bash
  # List all keys
  uv run python -m scripts.manage_tokens list

  # Revoke test keys
  uv run python -m scripts.manage_tokens revoke va_api_test...
  ```

- [ ] **AUTH_REQUIRED=true**: Authentication enabled (default in v1.3.0+)
  ```bash
  # Verify in environment
  echo $AUTH_REQUIRED  # Should be empty or "true"
  ```

- [ ] **Keys Stored Securely**: Using secrets manager, not hardcoded
  - AWS Secrets Manager
  - HashiCorp Vault
  - Azure Key Vault
  - Environment variables only (minimum security)

### 🌐 Network Security

- [ ] **HTTPS Enabled**: All traffic encrypted with TLS/SSL
  ```nginx
  # nginx example
  server {
      listen 443 ssl;
      ssl_certificate /path/to/cert.pem;
      ssl_certificate_key /path/to/key.pem;
      ssl_protocols TLSv1.2 TLSv1.3;

      location / {
          proxy_pass http://localhost:18011;
      }
  }
  ```

- [ ] **CORS Restricted**: Specific origins only, no wildcards
  ```bash
  export CORS_ORIGINS="https://app.yourdomain.com,https://admin.yourdomain.com"
  # Never: CORS_ORIGINS="*"
  ```

- [ ] **Firewall Rules**: API server not directly exposed to internet
  ```bash
  # Only allow reverse proxy
  sudo ufw allow from 127.0.0.1 to any port 18011
  sudo ufw deny 18011
  ```

- [ ] **Rate Limiting**: Prevent abuse (implement in reverse proxy)
  ```nginx
  # nginx rate limiting
  limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

  location /api/ {
      limit_req zone=api_limit burst=20 nodelay;
  }
  ```

### 🗄️ Data Protection

- [ ] **Database Encryption**: SQLite file encrypted or on encrypted volume
  ```bash
  # Use LUKS, dm-crypt, or cloud provider encryption
  ```

- [ ] **Token Storage Secure**: `tokens/tokens.json` has proper permissions
  ```bash
  chmod 600 tokens/tokens.json
  chmod 600 tokens/encryption.key
  chown videoannotator:videoannotator tokens/*
  ```

- [ ] **Storage Path Permissions**: Job storage not world-readable
  ```bash
  chmod 750 storage/
  chown -R videoannotator:videoannotator storage/
  ```

- [ ] **Backup Strategy**: Regular backups of database and tokens
  ```bash
  # Example backup script
  #!/bin/bash
  DATE=$(date +%Y%m%d_%H%M%S)
  tar -czf "backup_${DATE}.tar.gz" \
      videoannotator.db \
      tokens/ \
      storage/
  aws s3 cp "backup_${DATE}.tar.gz" s3://backups/videoannotator/
  ```

### 🔧 Configuration

- [ ] **DEBUG=false**: Debug mode disabled
  ```bash
  export DEBUG=false
  # Or remove DEBUG variable entirely
  ```

- [ ] **Secure Defaults**: Review all environment variables
  ```bash
  # .env.production
  AUTH_REQUIRED=true
  CORS_ORIGINS=https://app.yourdomain.com
  DEBUG=false
  LOG_LEVEL=INFO
  VIDEOANNOTATOR_STORAGE_DIR=/var/lib/videoannotator/storage
  ```

- [ ] **Log Sanitization**: No sensitive data in logs
  - API keys masked in logs
  - User emails partially redacted
  - File paths sanitized

- [ ] **Storage Retention**: Automatic cleanup configured
  ```python
  # config.yaml
  storage:
    retention_days: 90
    auto_cleanup: true
    cleanup_schedule: "0 2 * * *"  # 2 AM daily
  ```

### 📊 Monitoring & Logging

- [ ] **Centralized Logging**: Logs sent to monitoring service
  - Papertrail
  - Datadog
  - CloudWatch
  - ELK Stack

- [ ] **Security Alerts**: Monitor for suspicious activity
  ```yaml
  # Example alert rules
  - name: Failed Authentication Attempts
    condition: "count(401 responses) > 10 in 5 minutes"
    action: email_admin

  - name: Unusual API Usage
    condition: "requests per minute > 1000"
    action: email_admin
  ```

- [ ] **Access Logs**: All API requests logged with user context
  ```bash
  tail -f logs/api_requests.log
  ```

- [ ] **Error Monitoring**: Exception tracking enabled
  - Sentry
  - Rollbar
  - Bugsnag

### 🛡️ Server Hardening

- [ ] **Non-Root User**: Run as dedicated user account
  ```bash
  sudo useradd -r -s /bin/false videoannotator
  sudo chown -R videoannotator:videoannotator /opt/videoannotator
  ```

- [ ] **System Updates**: OS and dependencies up to date
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

- [ ] **Minimal Packages**: Only necessary software installed
  ```bash
  # Remove unnecessary packages
  sudo apt autoremove -y
  ```

- [ ] **SELinux / AppArmor**: Mandatory access control enabled
  ```bash
  # Check SELinux status
  sestatus

  # Or AppArmor
  sudo aa-status
  ```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile.production
FROM python:3.12-slim

# Run as non-root user
RUN useradd -r -u 1000 videoannotator

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=videoannotator:videoannotator . /app
WORKDIR /app

# Set secure permissions
RUN chmod 750 /app && \
    mkdir -p /app/storage /app/tokens /app/logs && \
    chown -R videoannotator:videoannotator /app

# Switch to non-root user
USER videoannotator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:18011/health || exit 1

# Start application
CMD ["uv", "run", "videoannotator"]
```

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  videoannotator:
    build:
      context: .
      dockerfile: Dockerfile.production
    restart: unless-stopped
    environment:
      - AUTH_REQUIRED=true
      - CORS_ORIGINS=https://app.yourdomain.com
      - DEBUG=false
      - LOG_LEVEL=INFO
    volumes:
      - ./storage:/app/storage:rw
      - ./tokens:/app/tokens:ro
      - ./logs:/app/logs:rw
    ports:
      - "127.0.0.1:18011:18011"  # Only localhost
    networks:
      - internal
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    networks:
      - internal
    depends_on:
      - videoannotator

networks:
  internal:
    driver: bridge
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: videoannotator
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: videoannotator
  template:
    metadata:
      labels:
        app: videoannotator
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: videoannotator
        image: videoannotator:1.3.0
        ports:
        - containerPort: 18011
        env:
        - name: AUTH_REQUIRED
          value: "true"
        - name: CORS_ORIGINS
          valueFrom:
            configMapKeyRef:
              name: videoannotator-config
              key: cors_origins
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: videoannotator-secrets
              key: api_key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 18011
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 18011
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: storage
          mountPath: /app/storage
        - name: tokens
          mountPath: /app/tokens
          readOnly: true
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: videoannotator-storage
      - name: tokens
        secret:
          secretName: videoannotator-tokens
      - name: tmp
        emptyDir: {}
```

## Post-Deployment Verification

### Security Testing

```bash
# 1. Verify HTTPS is enforced
curl -I http://yourdomain.com/api/v1/health
# Should redirect to HTTPS or return 301/302

# 2. Test authentication is required
curl http://yourdomain.com/api/v1/jobs
# Should return 401 Unauthorized

# 3. Test CORS restrictions
curl -H "Origin: http://evil.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS https://yourdomain.com/api/v1/jobs
# Should NOT include access-control-allow-origin header

# 4. Verify API key works
curl -H "Authorization: Bearer $API_KEY" \
     https://yourdomain.com/api/v1/jobs
# Should return 200 OK

# 5. Test rate limiting (if configured)
for i in {1..100}; do
  curl https://yourdomain.com/api/v1/health &
done
# Should eventually return 429 Too Many Requests
```

### Penetration Testing

Consider hiring security professionals to test:
- SQL injection attempts
- XSS vulnerabilities
- CSRF attacks
- Authentication bypass attempts
- Rate limit bypass
- File upload exploits

### Vulnerability Scanning

```bash
# Scan Docker image
docker scan videoannotator:1.3.0

# Scan dependencies
uv run pip-audit

# Check for CVEs
uv run safety check
```

## Ongoing Maintenance

### Weekly Tasks

- [ ] Review access logs for anomalies
- [ ] Check error rates in monitoring
- [ ] Verify backup completion

### Monthly Tasks

- [ ] Update dependencies
  ```bash
  uv sync --upgrade
  ```

- [ ] Review and rotate API keys
- [ ] Check disk space usage
- [ ] Review security alerts

### Quarterly Tasks

- [ ] Rotate API keys (90-day expiration)
- [ ] Update SSL certificates (if not automated)
- [ ] Review and update CORS origins
- [ ] Security audit
- [ ] Penetration testing

## Incident Response

### Security Breach Response

1. **Immediate Actions**:
   ```bash
   # Revoke all API keys
   uv run python -m scripts.manage_tokens revoke-all

   # Stop API server
   systemctl stop videoannotator

   # Review access logs
   grep -i "suspicious" logs/api_requests.log
   ```

2. **Investigation**:
   - Review all logs for unauthorized access
   - Check database for data exfiltration
   - Identify attack vector
   - Document timeline

3. **Recovery**:
   - Patch vulnerability
   - Generate new API keys
   - Notify affected users
   - Update security measures
   - Resume service

### Key Compromise

```bash
# 1. Immediately revoke compromised key
uv run python -m scripts.manage_tokens revoke va_api_compromised_key

# 2. Generate new keys
uv run python -m scripts.manage_tokens create \
  --user-id admin \
  --expires-in-days 90

# 3. Update all applications using the old key

# 4. Review logs for unauthorized usage
grep "va_api_compromised_key" logs/api_requests.log
```

## Compliance

### GDPR Considerations

- [ ] **Data Minimization**: Only store necessary data
- [ ] **Right to Erasure**: Implement data deletion
  ```bash
  uv run videoannotator job delete <job_id> --purge
  ```

- [ ] **Data Portability**: Export user data in standard format
- [ ] **Consent Management**: Track user consent for processing
- [ ] **Breach Notification**: 72-hour notification process

### HIPAA Considerations (if applicable)

- [ ] **BAA in Place**: Business Associate Agreement signed
- [ ] **Audit Logging**: All PHI access logged
- [ ] **Encryption at Rest**: Database and storage encrypted
- [ ] **Encryption in Transit**: HTTPS/TLS enabled
- [ ] **Access Controls**: Role-based access implemented

## Security Contacts

- **Security Issues**: security@yourdomain.com
- **Emergency Contact**: +1-XXX-XXX-XXXX
- **Incident Response Team**: incidents@yourdomain.com

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [VideoAnnotator Security Documentation](README.md)

## See Also

- [Authentication Guide](authentication.md) - API key management
- [CORS Configuration](cors.md) - Cross-origin security
- [Deployment Guide](../deployment/Docker.md) - Production deployment
