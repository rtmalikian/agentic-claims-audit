# SECURITY.md — honest limits

- **Synthetic data only.** The corpus and claims in this repo are generated. No PHI is included, requested, or needed to run anything here.
- **Proof of concept.** Not clinically tested, not a medical device, not a production audit system. See the disclaimer at the top of README.md.
- **No authentication** is implemented on the demo API in v0.1; do not expose it to a network with real data.
- The gateway's stub backend makes no external calls. If you plug in a real OpenAI-compatible upstream, claims data would leave your environment — do that only under your own HIPAA/BAA arrangements, which this repo does not provide.
- Findings are first-pass audit signals with citations, not payment decisions. Human-review routing exists because some categories must not auto-finalise.
- Report issues via GitHub issues on this repository.
