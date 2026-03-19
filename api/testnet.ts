import type { IncomingMessage, ServerResponse } from 'http';

const TARGET = 'https://rpc.testnet.aztec-labs.com';

export default async function handler(
  req: IncomingMessage,
  res: ServerResponse
) {
  const url = new URL(req.url ?? '/', `http://${req.headers.host}`);
  const subPath = url.pathname.replace(/^\/api\/testnet/, '') || '/';
  const targetUrl = `${TARGET}${subPath}${url.search}`;

  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (
      key.toLowerCase() !== 'host' &&
      key.toLowerCase() !== 'connection' &&
      value !== undefined
    ) {
      headers[key] = Array.isArray(value) ? value.join(', ') : value;
    }
  }

  const bodyChunks: Buffer[] = [];
  for await (const chunk of req) {
    bodyChunks.push(chunk as Buffer);
  }
  const body = bodyChunks.length ? Buffer.concat(bodyChunks) : undefined;

  const upstream = await fetch(targetUrl, {
    method: req.method,
    headers,
    body: body?.length ? body : undefined,
  });

  res.statusCode = upstream.status;

  const STRIP_HEADERS = new Set([
    'transfer-encoding',
    'content-encoding', // Node fetch auto-decompresses; re-sending this header causes ERR_CONTENT_DECODING_FAILED
    'content-length', // Byte length changes after decompression
  ]);
  upstream.headers.forEach((value, key) => {
    if (!STRIP_HEADERS.has(key.toLowerCase())) {
      res.setHeader(key, value);
    }
  });

  const responseBody = await upstream.arrayBuffer();
  res.end(Buffer.from(responseBody));
}
