from fastapi import FastAPI, Request
import socket

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    # Print all headers
    print("Request headers:")
    for name, value in request.headers.items():
        print(f"{name}: {value}")
    
    return {"message": "Hello from App 1", "hostname": socket.gethostname()}

@app.get("/health")
def health_check():
    return {"status": "ok"}