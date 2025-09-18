# BigQuery AI.GENERATE Connection Setup - Complete Guide

The error you're seeing indicates that the BigQuery connection's service account doesn't have proper permissions. Here's a complete step-by-step fix:

## **Step 1: Verify APIs are Enabled**

Make sure these APIs are enabled in your Google Cloud Console:
- BigQuery API
- BigQuery Connection API  
- Vertex AI API

Go to: **APIs & Services > Library** and search for each API to enable them.

## **Step 2: Create a New Cloud Resource Connection**

1. **Go to BigQuery Console**
2. **Click the "+ ADD" button** in the Explorer pane (left side)
3. **Select "Connection to external data sources"**
4. **Choose these settings:**
   - **Connection type**: "Vertex AI remote models, remote functions and BigLake (Cloud Resource)"
   - **Connection ID**: `vertex-ai-connection` (or your preferred name)
   - **Location**: `us` (must match your dataset location)
5. **Click "Create connection"**

## **Step 3: Get the Service Account ID**

1. **Click "Go to connection"** after creation
2. **In the Connection info pane**, copy the **Service account ID**
   - It will look like: `bqcx-xxxxxx-xxxx@gcp-sa-bigquery-condel.iam.gserviceaccount.com`
   - This should be: `bqcx-548059147147-4s97@gcp-sa-bigquery-condel.iam.gserviceaccount.com`

## **Step 4: Grant Vertex AI User Role**

### Option A: Using Google Cloud Console
1. **Go to IAM & Admin** in Google Cloud Console
2. **Click "Grant Access"**
3. **In "New principals" field**, enter the service account from Step 3
4. **Select Role**: `Vertex AI User` (roles/aiplatform.user)
5. **Click "Save"**

### Option B: Using gcloud command
```bash
gcloud projects add-iam-policy-binding trilink-471315 \
  --member='serviceAccount:bqcx-548059147147-4s97@gcp-sa-bigquery-condel.iam.gserviceaccount.com' \
  --role='roles/aiplatform.user'
```

## **Step 5: Wait for Permission Propagation**

**Wait 2-5 minutes** for the IAM changes to propagate across Google Cloud services.

## **Step 6: Update Your SQL Query**

Update your connection_id in the query to match your new connection:

```sql
connection_id => 'us.vertex-ai-connection'  -- Replace with your actual connection ID
```

## **Step 7: Verify Connection Details**

You can check your connection details with:

```sql
SELECT * FROM `trilink-471315.INFORMATION_SCHEMA.CONNECTIONS`
WHERE location = 'us'
```

## **Alternative Solution: Create Connection via SQL**

If the console method doesn't work, try creating the connection via SQL:

```sql
CREATE OR REPLACE EXTERNAL CONNECTION `us.vertex_ai_connection`
OPTIONS (
  type = 'CLOUD_RESOURCE',
  location = 'us'
);
```

Then repeat steps 3-6 with the new connection.

## **Troubleshooting Tips**

1. **Location Mismatch**: Ensure your connection location matches your dataset location (both should be 'us')

2. **Project Scope**: Make sure you're granting the Vertex AI User role in the correct project (`trilink-471315`)

3. **Service Account Format**: The service account should always be in format: `bqcx-xxxxx-xxx@gcp-sa-bigquery-condel.iam.gserviceaccount.com`

4. **Permission Timing**: IAM changes can take up to 5 minutes to propagate

5. **API Quotas**: Ensure your project has quota for Vertex AI API calls

## **Final Test Query**

After completing all steps, test with a simple query:

```sql
SELECT AI.GENERATE(
  'Hello, how are you?',
  connection_id => 'us.vertex-ai-connection',
  endpoint => 'gemini-2.5-flash'
).result AS test_message
```

If this works, your original complex query should also work!