"""Append after unchanged GPT-OSS compatibility definitions for a private matrix."""

from duma.integrations.kaggle import runner
from duma.integrations.kaggle.artifacts import read_checkpoint
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.utils.utils import DATA_DIR


def execute_matrix(llm, protocol, output):
    protocol.validate_data(DATA_DIR)
    rows = evaluation_rows(protocol, "dual")
    if len(rows) != 105 or protocol.trial_seeds != [300, 301, 302]:
        raise ValueError("Expected complete 105-row matrix")
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("*.json"):
        raw = json.loads(path.read_text())
        if read_checkpoint(path, raw["metadata"])["status"] != "completed":
            raise RuntimeError(
                "Existing unfinished row requires an explicit retry selection"
            )
    reports = output / "summaries"
    reports.mkdir(exist_ok=True)
    original_backend = runner.KaggleBackend
    create = llm.client.chat.completions.create
    stop, calls, errors = {}, [], []

    class ObservedBackend(KaggleBackend):
        def __call__(self, **kwargs):
            try:
                return super().__call__(**kwargs)
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code in {401, 403, 404, 429}:
                    stop.update(
                        http_status=code, role=self.role, error_type=type(exc).__name__
                    )
                raise

    def capture(*args, **kwargs):
        if args:
            raise ValueError("Expected keyword request")
        if kwargs.get("model") != llm.model:
            return create(**kwargs)
        changed = normalize_tools(kwargs)
        record = {
            "original_request_hash": hashlib.sha256(
                json.dumps(kwargs, sort_keys=True).encode()
            ).hexdigest(),
            "effective_request_hash": hashlib.sha256(
                json.dumps(changed, sort_keys=True).encode()
            ).hexdigest(),
        }
        try:
            response = create(**changed)
            record["choices"] = [
                {"finish_reason": c.finish_reason, "message_is_none": c.message is None}
                for c in response.choices or []
            ]
            return response
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
            raise
        finally:
            calls.append(record)

    llm.client.chat.completions.create = capture
    runner.KaggleBackend = ObservedBackend
    try:
        for row in rows:
            calls.clear()
            try:
                runner.run_episode(
                    llm,
                    protocol,
                    **{k: v for k, v in row.items() if k != "seed"},
                    output_dir=output,
                )
            except Exception as exc:
                errors.append(
                    {
                        **row,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1200],
                    }
                )
            wire_path = reports / (
                f"{row['domain']}-{row['task_id']}-{row['seed']}-wire.json"
            )
            if calls or not wire_path.exists():
                write_checkpoint(
                    wire_path,
                    {
                        "row": row,
                        "profile_hash": protocol.fingerprint,
                        "target_calls": list(calls),
                    },
                )
            print(
                json.dumps(
                    {**row, "attempted_target_calls": len(calls), "early_stop": stop}
                ),
                flush=True,
            )
            if stop:
                break
    finally:
        runner.KaggleBackend = original_backend
        llm.client.chat.completions.create = create
    items = []
    for path in output.glob("*.json"):
        raw = json.loads(path.read_text())
        items.append(read_checkpoint(path, raw["metadata"]))
    summary = aggregate(items, protocol, llm.model, "dual")
    report = {
        **summary,
        "profile": protocol.model_dump(),
        "profile_hash": protocol.fingerprint,
        "early_stop": stop,
        "errors": errors,
        "scope": "GPT-OSS low plus named-empty-descriptions profile, not frozen V4; strict original scoring, no safe-block adjudication",
    }
    write_checkpoint(reports / "matrix.json", report)
    return {k: v for k, v in report.items() if k not in {"profile", "errors"}}


@kbench.task(name="duma-gptoss-compatible-dual-v1")
def full_dual(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "openai/gpt-oss-120b":
        return {"skipped": "Target-only readiness, no inference or security score"}
    kbench.config.interactive_mode = False
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol = make_protocol(base, GPTOSS_ADAPTER_SHA256)
    if (
        protocol.fingerprint
        != "7792ba790baee00c290b213c431203b4f969241e25e174a44764326c53dc0315"
    ):
        raise ValueError("Full profile differs from verified preflight")
    return execute_matrix(llm, protocol, Path("/kaggle/working/duma-kaggle-results"))


if __name__ == "__main__":
    full_dual.run(kbench.llm)
