"""
Local Webhook Test Server
Start this alongside ngrok to receive Freshdesk webhooks locally.

Usage:
    1. Start this server:     python run_local_server.py
    2. Start ngrok:           ngrok http 8000
    3. Configure Freshdesk webhook with ngrok URL
    4. Create a ticket in Freshdesk - workflow runs automatically!
"""

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       🌊 FLUSSO LOCAL WEBHOOK SERVER                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║   Server starting on http://localhost:8000                ║
    ║                                                           ║
    ║   NEXT STEPS:                                             ║
    ║   1. Open new terminal and run: ngrok http 8000           ║
    ║   2. Copy the https://xxx.ngrok.io URL                    ║
    ║   3. Add webhook in Freshdesk Admin → Automations         ║
    ║      URL: https://xxx.ngrok.io/freshdesk/webhook          ║
    ║   4. Create a test ticket - watch the magic happen!       ║
    ║                                                           ║
    ║   Endpoints:                                              ║
    ║   • GET  /              - Health check                    ║
    ║   • GET  /health        - Quick health                    ║
    ║   • POST /freshdesk/webhook  - Ticket webhook             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
