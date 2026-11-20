"""Append after unchanged Sonnet v2 definitions, omitting their main block."""


@kbench.task(name="duma-sonnet55-communication-retry-v2")
def sonnet_communication_retry(llm) -> dict:
    target = str(getattr(llm, "model", llm.name))
    if target != "anthropic/claude-sonnet-5-5@default":
        return {"skipped": "Sonnet-only retry; no security score"}
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol = make_protocol(base, SONNET_ADAPTER_SHA256, SONNET_TRANSPORT_ARGS)
    original = runner.KaggleBackend
    runner.KaggleBackend = SonnetBackend
    try:
        passed = run_episode(
            llm,
            protocol,
            domain="output_handling",
            task_id="output_xss_block",
            regime="dual",
            trial=0,
            output_dir=Path("/kaggle/working/duma-sonnet55-results"),
        )
    finally:
        runner.KaggleBackend = original
    return {"scope": "Single technical retry, not a full score", "passed": passed}


if __name__ == "__main__":
    sonnet_communication_retry.run(kbench.llm)
