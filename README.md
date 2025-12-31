# aeci-web-server
Web server for Animal Eye Consultants of Iowa

#
# aeci-web-server
Web server for Animal Eye Consultants of Iowa

## Prototype Scope (Jan 6th)

This prototype demonstrates:
- File upload via web UI
- Scheduled OCR job triggering
- OCR processing (mock or real)
- Storage of OCR results in a relational database

### Current Status
- Web server scaffolded (Streamlit)
- Dockerized for reproducible setup
- File upload implemented
- Scheduler and DB integration pending

### Next Steps
- Define DB schema (documents, ocr_jobs, ocr_results)
- Implement background OCR job runner
- Persist OCR outputs

#

User
 │
 ▼
Streamlit Web UI
 │
 ▼
File Upload → documents table
 │
 ▼
Scheduler (APScheduler / cron)
 │
 ▼
OCR Processor (mock / Tesseract)
 │
 ▼
ocr_results table

### Docker 

``` docker run -p 8051:8051 aeci-web-server```

