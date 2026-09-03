# ComfyUI MiniMax H3 — Audio References

Three ComfyUI nodes that let you feed a **reference audio** to MiniMax H3, alongside the stock nodes.

MiniMax H3's `ref2va` task supports reference conditions (images, videos, audio). Its `fl2va` task supports first/last keyframes. Under the hood both are the same mechanism — extra rows in the packed sequence that get re-injected at every sampling step and are never denoised — so an audio reference can ride along with the stock **MiniMax H3 Image to Video** node.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rauldlnx10/ComfyUI-MiniMaxH3-Separate.git
```

Restart ComfyUI. No extra dependencies — everything used ships with ComfyUI.

## The nodes

### MiniMax H3 Ref Audio
Encodes an audio clip into a reference block.

| Input | Notes |
|---|---|
| `audio_vae` | The **audio** VAE (`minimax_h3_audio_vae`), not the video one |
| `audio` | From `LoadAudio` or any AUDIO source |
| `max_seconds` | Trim the reference. `0` = use it whole |
| `refs` | Chain input, to stack several references |

`max_seconds` matters: every second of reference is 40 latent frames = 80 rows in the packed sequence, and those rows pass through every sampling step. Start at 2–3 seconds.

### MiniMax H3 Apply Refs
Attaches the reference blocks to the conditioning. Goes right after your conditioning node.

### MiniMax H3 Fix Cond Payload
Goes on the model. **Required whenever you combine keyframes with references.**

`comfy/model_base.py` builds the DiT payload with two independent branches, and the refs branch overwrites the keyframes':

```python
if keyframes: payload["cond_video_latents"] = [kf["latent"] ...]
if refs:      payload["cond_video_latents"] = [r["latent"] ...]   # clobbers
```

The packed layout reserves rows for **both** (keyframe `cond` segments first, then the refs) and the forward pass fills them in that order from this single list. With an audio-only reference the second branch leaves the list empty, so the keyframe rows get nothing while the layout still holds their space.

This node installs a `DIFFUSION_MODEL` wrapper that rebuilds the list in layout order, just before the forward pass. It is harmless when only one of the two is in use.

## Usage

```
LoadImage (first/last) ─┐
CLIPLoader ─────────────┼─> MiniMax H3 Image to Video ─> positive ──┐
VAE (video) ────────────┘                    └─> LATENT ─> Sampler  │
                                                                    │
LoadAudio ─> Ref Audio ─> Apply Refs ◄──────────────────────────────┘
   VAE (audio) ─┘              └─> BasicGuider

UNETLoader ─> ModelSamplingMiniMaxH3 ─> Fix Cond Payload ─> guider + scheduler
```

An example workflow is in [`workflows/`](workflows/).

## Known limitation

The stock node tokenizes the prompt through the `images=` path, which emits no `<Audio 1>` label. The model receives the reference rows, but the presentation text never names them. The `ref2va` presentation does emit that label; reproducing it here would mean replacing the text encode entirely.

## Notes

- `fl2va` was never trained with audio references — that's the `ref2va` checkpoint. It may ignore the reference, or degrade the generated audio. Treat this as an experiment.
- The reference is resampled to the audio VAE's rate automatically.
- To disable the audio branch, bypass (Ctrl+B) **Ref Audio** and **Apply Refs**. The rest of the graph keeps working.

## Requirements

ComfyUI with native MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`), plus the H3 model files: a diffusion checkpoint, the Qwen3-VL text encoder, and both the video and audio VAEs.
