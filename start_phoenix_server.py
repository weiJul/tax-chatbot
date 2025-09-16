#!/usr/bin/env python3
"""
Standalone Phoenix-Arize AI server for tax RAG evaluation dashboard
Keep this running to access the evaluation dashboard at http://localhost:6006
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import phoenix as px
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_phoenix_server():
    """Start persistent Phoenix server"""
    try:
        print("🚀 Starting Phoenix-Arize AI Dashboard Server...")
        print("=" * 50)
        
        # Launch Phoenix with environment variables (recommended way)
        import os
        os.environ['PHOENIX_PORT'] = '6006'
        os.environ['PHOENIX_HOST'] = 'localhost'
        
        # Start Phoenix session
        session = px.launch_app()
        
        print("✅ Phoenix server started successfully!")
        print("🌍 Dashboard URL: http://localhost:6006")
        print("📊 Available features:")
        print("   • Real-time performance metrics")
        print("   • Query tracing and analysis") 
        print("   • Tax accuracy evaluation")
        print("   • Hallucination detection")
        print("   • Jurisdiction classification")
        print("   • System health monitoring")
        print()
        print("💡 To generate evaluation data, run in another terminal:")
        print("   python demo_phoenix_evaluation.py")
        print()
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Phoenix server...")
            if hasattr(session, 'close'):
                session.close()
            print("✅ Phoenix server stopped")
            
    except Exception as e:
        print(f"❌ Failed to start Phoenix server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_phoenix_server()