# Keep Render Service Alive - Setup Guide

## Option 1: GitHub Actions (RECOMMENDED - FREE)
The workflow file (.github/workflows/keepalive.yml) is already configured.
It will automatically ping your service every 10 minutes.

**What it does:**
- Runs on GitHub's free runners every 10 minutes
- Sends GET request to /healthz endpoint
- Keeps your Render service from sleeping

**No setup needed!** Just push to GitHub and it will work.

## Option 2: External Cron Service (UptimeRobot - FREE)

1. Go to https://uptimerobot.com
2. Sign up (free account)
3. Click "Add New Monitor"
4. Configure:
   - Monitor Type: HTTP(s)
   - URL: https://quiz-solver-agent.onrender.com/healthz
   - Monitoring Interval: 5 minutes
   - Click "Create Monitor"

**Benefits:**
- Very reliable
- Shows uptime statistics
- Email alerts if service goes down
- Free tier monitors up to 50 services

## Option 3: Cron-Job.Org (FREE)

1. Go to https://cron-job.org
2. Sign up (free)
3. Click "Create Cronjob"
4. Configure:
   - URL: https://quiz-solver-agent.onrender.com/healthz
   - Execution times: Every 5 minutes
   - Click "Create"

**Benefits:**
- Completely free
- Simple interface
- Reliable execution

## Option 4: Render Pro Tier (PAID)

If you upgrade to Pro plan ($12/month):
- Service never sleeps
- Always running
- Best for production

**Choose Pro if:**
- You need guaranteed uptime
- Service should be available 24/7
- You have budget for it

## Current Setup

Your app has a `/healthz` endpoint that returns:
```json
{
  "status": "ok",
  "uptime_seconds": 12345
}
```

This is perfect for keepalive monitoring!

## Testing Locally

Test your keepalive endpoint:
```powershell
curl http://localhost:8000/healthz
```

Test on Render (once deployed):
```powershell
curl https://quiz-solver-agent.onrender.com/healthz
```

## Recommendation

1. **Start with GitHub Actions** (already configured, no extra service)
2. **If that fails, use UptimeRobot** (very reliable, free)
3. **Upgrade to Render Pro** when you have budget

The keepalive will prevent your service from being spun down after 15 minutes of inactivity!
