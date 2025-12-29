import logging
import time
from typing import Dict, Tuple, List

import websockets.sync.client

from . import msgpack_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        max_retries = 10
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                conn = websockets.sync.client.connect(
                    self._uri, 
                    compression=None, 
                    max_size=None, 
                    open_timeout=10,
                    ping_timeout=1e3, 
                    close_timeout=1e3
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logger.info(f"Connected to server. Metadata: {metadata}")
                return conn, metadata
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to server at {self._uri}")

    def infer(self, obs: List[Dict]) -> Dict:
        """
        Send observation(s) to server and get action.
        
        Args:
            obs: List of observation dictionaries or single observation dict
            
        Returns:
            Action dictionary
        """
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        """Reset the policy on the server."""
        reset_msg = [{"mode": "reset"}]
        self.infer(reset_msg)
    
    def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            self._ws.close()
            logger.info("WebSocket connection closed")