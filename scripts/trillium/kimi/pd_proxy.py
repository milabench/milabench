"""K3-XN part 3: vLLM v0.32.0's NIXL toy proxy, unchanged, plus GET /v1/models, which milabench's readiness check polls.

The proxy (tests/v1/kv_connector/nixl_integration/toy_proxy_server.py at v0.32.0, downloaded beside the launcher, see
pd_serve.sh) sends each completion to a prefill instance with max_tokens=1 and do_remote_decode, then to a decode instance
with the kv_transfer_params the prefill returned. /v1/models answers once every instance answers its own.
--colocated: the colocated reference of the P/D runs, through the same proxy: no prefill call, so each request runs whole
on the next decode instance, round-robin (pd_serve.sh passes every instance as a decoder).

    python pd_proxy.py [--colocated] <toy_proxy_server.py's arguments>
"""

import sys

import uvicorn
from fastapi import Response

import toy_proxy_server as proxy


@proxy.app.get("/v1/models")
async def models():
    body = None
    for info in proxy.app.state.prefill_clients + proxy.app.state.decode_clients:
        try:
            response = await info["client"].get("/models")
        except Exception:
            return Response(status_code=503)
        if response.status_code != 200:
            return Response(status_code=503)
        body = response.json()
    return body


class _NoPrefill:
    """What the proxy reads of a prefill response: no kv_transfer_params, so the decode instance does the whole request"""

    def json(self):
        return {}

    async def aclose(self):
        pass


async def _no_prefill(*args, **kwargs):
    return _NoPrefill()


if __name__ == "__main__":
    if "--colocated" in sys.argv:
        sys.argv.remove("--colocated")
        proxy.send_request_to_service = _no_prefill
        print("[pd_proxy] colocated: requests go whole to the decoders, round-robin", file=sys.stderr, flush=True)
    proxy.global_args = proxy.parse_args()
    print(f"[pd_proxy] prefill {proxy.global_args.prefiller_instances} decode {proxy.global_args.decoder_instances}",
          file=sys.stderr, flush=True)
    uvicorn.run(proxy.app, host=proxy.global_args.host, port=proxy.global_args.port)
