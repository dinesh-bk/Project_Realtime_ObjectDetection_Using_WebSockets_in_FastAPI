import contextlib
import asyncio
import io
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from models import object_detection
from PIL import Image



@contextlib.asynccontextmanager
async def lifespan(app:FastAPI):
    object_detection.load_model()
    yield


app = FastAPI(lifespan=lifespan)

# Back pressure is a phenomenon that occurs in systems where there's a flow of data or tasks from one component to another, and the receiving component or system is unable to process or handle the incoming data or tasks at the same rate they are being sent. Here, we'll receive more images from the browser than the server is able to handle because of the time needed to run the detection algorithm which results in backpressure. Thus, we'll have to work with a queue(or buffer) of limited size and drop some images along the way to handle the stream in real time.


# First Task: Receive Bytes
# asyncio.Queue queue data in memory and retrieve it in FIFO strategy
async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await websocket.receive_bytes()
        try:
            # Two methods of queue:
            # put: If queue is full wait until there is room in the queue
            # put_nowait: does same like put but If queue is full exception: asyncio.QueueFull is raised
            queue.put_nowait(bytes)
        # If exception is raised we just pass and drop the data
        except asyncio.QueueFull:
            pass

# Second Task: Detect
async def detect(websocket: WebSocket, queue: asyncio.Queue):
    # Continuously loop to process incoming data
    while True:
        # Retrieve bytes data from the queue asynchronously
        bytes = await queue.get()
        
        # Convert the received bytes into an image using BytesIO and Image.open
        image = Image.open(io.BytesIO(bytes))
        
        # Perform object detection on the image using an object_detection instance
        objects = object_detection.predict(image)
        
        # Send the detected objects as a JSON message via the WebSocket
        await websocket.send_json(objects.dict())


# WebSocket route for handling object detection
@app.websocket("/object-detection")
async def ws_object_detection(websocket: WebSocket):
    # Accept the WebSocket connection
    await websocket.accept()
    
    # Create a queue with a maximum size of 1
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    
    # Create tasks to concurrently handle receiving and detecting
    receive_task = asyncio.create_task(receive(websocket, queue))
    detect_task = asyncio.create_task(detect(websocket, queue))
    
    try:
        # Wait for either 'receive_task' or 'detect_task' to complete
        done, pending = await asyncio.wait(
            {receive_task, detect_task}, 
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any tasks that are still pending
        for task in pending:
            task.cancel()
        
        # Get the result of each completed task
        for task in done:
            task.result()
    except WebSocketDisconnect:
        # Handle WebSocket disconnection, if it occurs
        pass

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent/"index.html")

static_files_app = StaticFiles(directory=Path(__file__).parent/"assets")
app.mount("/assets", static_files_app)