"""
gRPC block fetcher — subscribes to full confirmed blocks
Two modes: fetch_slot_range(start, end) and follow_confirmed()
"""

import asyncio
import grpc
import logging

import geyser_pb2
import geyser_pb2_grpc

logger = logging.getLogger(__name__)

CHANNEL_OPTIONS = [
    ('grpc.keepalive_time_ms', 10000),
    ('grpc.keepalive_timeout_ms', 3000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.http2.min_ping_interval_without_data_ms', 300000),
    ('grpc.max_receive_message_length', 64 * 1024 * 1024),
    ('grpc.max_send_message_length', 64 * 1024 * 1024),
]


class BlockFetcher:
    def __init__(self, grpc_endpoint: str, grpc_token: str = ''):
        self.grpc_endpoint = grpc_endpoint
        self.grpc_token = grpc_token

    def _create_channel(self):
        if 'localhost' in self.grpc_endpoint or ':443' not in self.grpc_endpoint:
            return grpc.aio.insecure_channel(self.grpc_endpoint, options=CHANNEL_OPTIONS)
        return grpc.aio.secure_channel(
            self.grpc_endpoint,
            grpc.ssl_channel_credentials(),
            options=CHANNEL_OPTIONS,
        )

    def _metadata(self):
        metadata = []
        if self.grpc_token:
            metadata.append(('x-token', self.grpc_token))
        return metadata

    async def _subscribe_blocks(self, stub, metadata, from_slot=None):
        """Create a block subscription request and return the stream."""
        request = geyser_pb2.SubscribeRequest(
            commitment=geyser_pb2.CommitmentLevel.CONFIRMED,
        )

        # Subscribe to full blocks with transactions
        block_filter = request.blocks["confirmed_blocks"]
        block_filter.include_transactions = True
        block_filter.include_accounts = False
        block_filter.include_entries = False

        if from_slot is not None:
            request.from_slot = from_slot

        stream = stub.Subscribe(iter([request]), metadata=metadata)
        return stream

    async def follow_confirmed(self):
        """Follow new confirmed blocks indefinitely, yielding (slot, block) tuples.
        Reconnects automatically on error."""
        retry_count = 0
        max_retries = 10

        while True:
            channel = self._create_channel()
            try:
                stub = geyser_pb2_grpc.GeyserStub(channel)
                metadata = self._metadata()

                # Test connection
                version_resp = await stub.GetVersion(geyser_pb2.GetVersionRequest(), metadata=metadata)
                logger.info(f"Connected to Geyser v{version_resp.version}")
                retry_count = 0

                stream = await self._subscribe_blocks(stub, metadata)
                async for update in stream:
                    if update.HasField('block'):
                        yield update.block.slot, update.block

            except grpc.aio.AioRpcError as e:
                retry_count += 1
                wait_time = min(retry_count * 2, 30)
                logger.error(f"gRPC error: {e.code()} - {e.details()}, retry in {wait_time}s ({retry_count}/{max_retries})")
                if retry_count >= max_retries:
                    raise RuntimeError("Failed to connect after max retries")
                await asyncio.sleep(wait_time)
            except Exception as e:
                retry_count += 1
                logger.error(f"Unexpected error: {e}, retry in 5s ({retry_count}/{max_retries})")
                if retry_count >= max_retries:
                    raise
                await asyncio.sleep(5)
            finally:
                await channel.close()

    async def fetch_slot_range(self, start_slot: int, end_slot: int):
        """Fetch blocks in a slot range, yielding (slot, block) tuples.
        Stops after processing end_slot."""
        channel = self._create_channel()
        try:
            stub = geyser_pb2_grpc.GeyserStub(channel)
            metadata = self._metadata()

            version_resp = await stub.GetVersion(geyser_pb2.GetVersionRequest(), metadata=metadata)
            logger.info(f"Connected to Geyser v{version_resp.version}")
            logger.info(f"Fetching slots {start_slot} to {end_slot}")

            stream = await self._subscribe_blocks(stub, metadata, from_slot=start_slot)
            async for update in stream:
                if update.HasField('block'):
                    slot = update.block.slot
                    if slot > end_slot:
                        logger.info(f"Reached end slot {end_slot}, stopping")
                        break
                    yield slot, update.block
        finally:
            await channel.close()
