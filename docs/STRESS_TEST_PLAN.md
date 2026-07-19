# Stress Test Plan

1. **Upload Spikes**: Flood the upload endpoint with 100 concurrent 50MB PDFs.
2. **Dojo Abuse**: Send 10,000 concurrent valid and invalid Python scripts.
3. **Database Concurrency**: Attempt 5,000 simultaneous writes to `LearnerProgress`.
