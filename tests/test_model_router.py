import unittest
from types import SimpleNamespace

from hypermindlabs import model_router as mr


class FakeAsyncClient:
    model_lists = {}
    list_errors = {}
    chat_actions = {}

    def __init__(self, host):
        self.host = host

    async def list(self):
        if self.host in self.list_errors:
            raise self.list_errors[self.host]
        models = [SimpleNamespace(model=name) for name in self.model_lists.get(self.host, [])]
        return SimpleNamespace(models=models)

    async def chat(self, model, messages, stream=False, **kwargs):
        action = self.chat_actions.get((self.host, model, stream))
        if isinstance(action, Exception):
            raise action
        if action is None:
            if stream:
                async def _default_stream():
                    yield {"message": {"content": "default"}}

                return _default_stream()
            return {"message": {"content": "default"}}
        if stream and callable(action):
            return action()
        return action


class TestModelRouter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_async_client = mr.AsyncClient
        mr.AsyncClient = FakeAsyncClient
        FakeAsyncClient.model_lists = {}
        FakeAsyncClient.list_errors = {}
        FakeAsyncClient.chat_actions = {}

    def tearDown(self):
        mr.AsyncClient = self.original_async_client

    def test_host_precedence_prefers_endpoint_override(self):
        router = mr.ModelRouter(
            inference_config={"chat": {"url": "http://configured:11434", "model": "m-config"}},
            endpoint_override="http://custom:11434",
        )
        self.assertEqual(router.resolve_host("chat"), "http://custom:11434")

    def test_default_local_host_used_when_not_configured(self):
        router = mr.ModelRouter(inference_config={})
        self.assertEqual(router.resolve_host("chat"), mr.DEFAULT_OLLAMA_HOST)

    def test_candidate_model_order_and_dedupe(self):
        router = mr.ModelRouter(
            inference_config={"chat": {"url": "http://configured:11434", "model": "m-config"}}
        )
        candidates = router.candidate_models(
            capability="chat",
            requested_model="m-requested",
            allowed_models=["m-requested", "m-fallback", "m-config"],
        )
        self.assertEqual(candidates, ["m-requested", "m-fallback", "m-config"])

    async def test_non_stream_chat_falls_back_to_next_model(self):
        host = "http://custom:11434"
        FakeAsyncClient.model_lists[host] = ["m1", "m2"]
        FakeAsyncClient.chat_actions[(host, "m1", False)] = RuntimeError("m1 failed")
        FakeAsyncClient.chat_actions[(host, "m2", False)] = {"message": {"content": "m2 ok"}}

        router = mr.ModelRouter(
            inference_config={"chat": {"url": host, "model": "m1"}},
            endpoint_override=host,
        )

        response, metadata = await router.chat_with_fallback(
            capability="chat",
            requested_model="m1",
            allowed_models=["m1", "m2"],
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

        self.assertEqual(response["message"]["content"], "m2 ok")
        self.assertEqual(metadata["selected_model"], "m2")
        self.assertEqual(metadata["attempted_models"], ["m1", "m2"])
        self.assertEqual(metadata["fallback_count"], 1)

    async def test_stream_chat_falls_back_and_returns_stream(self):
        host = "http://custom:11434"
        FakeAsyncClient.model_lists[host] = ["m1", "m2"]
        FakeAsyncClient.chat_actions[(host, "m1", True)] = RuntimeError("m1 stream failed")

        async def _stream_success():
            yield {"message": {"content": "chunk-1"}}
            yield {"message": {"content": "chunk-2"}}

        FakeAsyncClient.chat_actions[(host, "m2", True)] = _stream_success

        router = mr.ModelRouter(
            inference_config={"chat": {"url": host, "model": "m1"}},
            endpoint_override=host,
        )

        stream, metadata = await router.chat_with_fallback(
            capability="chat",
            requested_model="m1",
            allowed_models=["m1", "m2"],
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(metadata["selected_model"], "m2")
        self.assertEqual(metadata["attempted_models"], ["m1", "m2"])
        self.assertEqual(metadata["fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
