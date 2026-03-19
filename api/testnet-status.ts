import type { IncomingMessage, ServerResponse } from 'http';

const TARGET = 'https://rpc.testnet.aztec-labs.com/status';

export default async function handler(
  _req: IncomingMessage,
  res: ServerResponse
) {
  const upstream = await fetch(TARGET, { method: 'GET' });

  res.statusCode = upstream.status;

  const STRIP_HEADERS = new Set([
    'transfer-encoding',
    'content-encoding',
    'content-length',
  ]);
  upstream.headers.forEach((value, key) => {
    if (!STRIP_HEADERS.has(key.toLowerCase())) {
      res.setHeader(key, value);
    }
  });

  const body = await upstream.arrayBuffer();
  res.end(Buffer.from(body));
}
