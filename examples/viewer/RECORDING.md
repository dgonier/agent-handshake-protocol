# Recording the AHP walkthrough

A 90-second Loom that shows the protocol doing something real. Three
beats: **agents arguing** → **the receipts** → **the menu it cost
them**. Optimized for a README hero asset, not a tutorial.

The viewer is mobile-first, so record at phone resolution (390×844 in
Loom's device frame) — it reads better on a project page than a wide
desktop screenshot.

## Before you record

1. From the repo root, bring the stack up against a real Redis:

   ```bash
   cd examples/viewer
   docker compose up --build
   ```

2. (Optional — only if you want the Modal leaf to appear live on the
   menu during the recording.) Set the env vars in a `.env` next to
   `docker-compose.yml`:

   ```env
   AHP_MODAL_VLLM_URL=https://YOURUSER--vllm-qwen.modal.run/v1
   AHP_MODAL_VLLM_PROVE_HEALTH=1
   AHP_MODAL_VLLM_MODEL=qwen2-5-7b
   ```

   The default leaves Modal as a metadata-only entry, which still
   shows on the page but isn't routable. With `PROVE_HEALTH=1` the
   first run probes `GET {URL}/models` and flips the leaf live.

3. In a separate terminal, watch the audit log:

   ```bash
   docker compose logs -f viewer | grep -E "broker|settle|hold"
   ```

   This gives Loom narration anchors — "watch the broker place a hold,
   dispatch, settle on response."

4. Open `http://localhost:8000` on a phone or in a phone-sized window.
   Tabs you'll need open: `/`, `/economy`, `/audit`.

## Recording script

### Beat 1 — kick off the run (0:00–0:25)

* On `/`, fill in: topic ("**Is dark matter modified gravity in
  disguise?**"), keep org/domain/subdomain defaults, count = **3**.
* Switch format to `interview-yall` for variety, or stick with
  `debate`.
* Hit submit. Narration: *"Three agents get invited at runtime — the
  inviter LLM picks personas, the factory builds them, the broker
  funds their wallets."*

### Beat 2 — the receipts (0:25–0:50)

* While the run is going, switch to `/economy`.
* Scroll through:
  * **Wallets** — caller, agents, server, broker, commons all have
    visible balances.
  * **Servers** — one entry for the self-hosted org, plus
    `beta`/`modal-vllm` if you wired the Modal env. Narration: *"Two
    servers, each bound to a different compute backend, both registered
    with the broker as the source of truth."*

### Beat 3 — settlements + menu (0:50–1:30)

* Wait for the run to finish (browser status bar drops "running" and
  the wallet balances refresh).
* On `/economy`, scroll to **Recent settlements**. Narration: *"Each
  row is one agent response. The broker tracked latency, character
  count, picked the right compute leaf, and split the payment four
  ways: agent earns, compute provider gets its slice, broker tax,
  commons tax."*
* Scroll to **Compute menu**. Narration: *"This is the menu that was
  available at dispatch time. Self-hosted Bedrock at zero compute
  cost; the Modal vLLM leaf priced per character. Servers choose by
  pattern; the broker enforces health-proof + reputation."*
* End on a single still of the settlements table.

## Editing notes

* No need to show terminal output mid-recording — narration over the
  UI is enough. If you do want a terminal beat, slice in the `docker
  compose logs` window for ~2 seconds.
* The most legible single still is the settlements table, not the
  transcript. Save that frame for the README thumbnail.

## Putting it in the README

Once recorded, drop the Loom share link near the top of the repo
`README.md` (in the "Proof" section) with a one-line caption:

> Watch the protocol settle a real multi-agent debate in 90 seconds:
> [loom.com/share/…](https://loom.com/…)

If you want a still as a fallback in case the Loom link rots, pull a
frame from the recording, save as `docs/img/economy-settlements.png`,
and reference it as a fallback below the link.
