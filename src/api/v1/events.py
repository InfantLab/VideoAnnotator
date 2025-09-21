"""
Server-Sent Events endpoints for VideoAnnotator API
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio
import logging

router = APIRouter()

logger = logging.getLogger(__name__)


async def event_stream() -> AsyncGenerator[str, None]:
    """
    Generate server-sent events stream.
    
    This is a basic implementation that sends periodic heartbeat events.
    In v1.3.0, this will be enhanced with real-time job status updates.
    """
    try:
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': str(asyncio.get_event_loop().time())})}\n\n"
        
        # Send periodic heartbeat events
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            heartbeat_data = {
                'type': 'heartbeat',
                'timestamp': str(asyncio.get_event_loop().time()),
                'message': 'Server is alive'
            }
            yield f"data: {json.dumps(heartbeat_data)}\n\n"
            
    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by client")
        return
    except Exception as e:
        logger.error(f"Error in SSE stream: {e}")
        error_data = {
            'type': 'error',
            'message': str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/stream")
async def events_stream():
    """
    Server-Sent Events endpoint for real-time updates.
    
    This endpoint provides a basic SSE implementation for client compatibility.
    Enhanced real-time job status updates will be added in v1.3.0.
    
    Returns:
        StreamingResponse: SSE stream with periodic heartbeat events
    """
    logger.info("SSE client connected to /api/v1/events/stream")
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )