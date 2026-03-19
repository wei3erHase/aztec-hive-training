import type { IncomingMessage, ServerResponse } from 'http';

const TARGET = 'https://rpc.testnet.aztec-labs.com/status';

export default async function handler(
  _req: IncomingMessage,
  res: ServerResponse
) {
  const upstream = await fetch(TARGET, { method: 'GET' });

  res.statusCode = upstream.status;
  upstream.headers.forEach((value, key) => {
    if (key.toLowerCase() !== 'transfer-encoding') {
      res.setHeader(key, value);
    }
  });

  const body = await upstream.arrayBuffer();
  res.end(Buffer.from(body));
}
