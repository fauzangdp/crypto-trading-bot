import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://localhost:8766"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
            # Send ping
            await websocket.send("ping")
            print("📤 Sent ping")
            
            # Wait for pong
            response = await websocket.recv()
            print(f"📥 Received: {response}")
            
            # Request initial data
            await websocket.send(json.dumps({"type": "request_initial_data"}))
            print("📤 Requested initial data")
            
            # Wait for data
            data = await websocket.recv()
            parsed = json.loads(data)
            print(f"📥 Received data type: {parsed.get('type')}")
            
            print("✅ WebSocket test successful!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())