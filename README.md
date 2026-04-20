# Timesheet Approval POC

Single-app POC with:

- Step 1 details + date-wise hours
- Step 2 upload (PDF/image) auto extraction + normalization
- Step1 vs extracted comparison with green/red highlights
- Auto-approve/manual-review/trusted-template logic
- Manual decision streak tracking in SQLite
- AWS/Textract/Bedrock error visibility in UI
- In-memory file processing (no upload folder persistence)

## Local run

```bash
cd "d:\AWS\AWS Projects\SliceHRMS\TimesheetApproval_V4"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in project root:

```bash
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=<your-bedrock-model-id>
```

Set AWS credentials in terminal:

```bash
AWS_ACCESS_KEY_ID=<...>
AWS_SECRET_ACCESS_KEY=<...>
AWS_SESSION_TOKEN=<...>   # if temp credentials
```

Run:

```bash
streamlit run app.py
```

Notes:
- Uploaded files are processed from memory and not written to an uploads directory.
- If token/model access is invalid, the exact AWS error is shown in the app.
