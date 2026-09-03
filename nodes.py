"""Audio references for MiniMax H3, usable with the stock nodes.

The ref2va task feeds the DiT reference conditions; fl2va feeds keyframes.
Structurally they are the same mechanism - extra rows in the packed sequence
that are re-injected every step and never denoised - so an audio reference can
ride alongside the stock `MiniMax H3 Image to Video` node.

    LoadAudio -> MiniMaxH3RefAudio -> MiniMaxH3ApplyRefs -> sampler
                                                ^
    MiniMax H3 Image to Video (positive) -------/

    model -> MiniMaxH3FixCondPayload -> guider

The one thing that does NOT come for free: the stock node tokenizes the prompt
through the `images=` path, which emits no "<Audio 1>" label. The model gets the
reference rows without the presentation text naming them.
"""

import torchaudio
import comfy.patcher_extension
import node_helpers
from comfy_api.latest import ComfyExtension, io

# chainable list of DiT ref blocks
MiniMaxRefs = io.Custom("MINIMAX_H3_REFS")


class MiniMaxH3RefAudio(io.ComfyNode):
    """Encode a reference audio into a DiT ref block."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefAudio",
            display_name="MiniMax H3 Ref Audio",
            category="model/conditioning/minimax",
            description="Reference audio condition. Native to the ref2va checkpoint; on fl2va it is experimental - that model never saw audio references in training. Every second of reference adds 40 latent frames = 80 rows to the packed sequence, and those rows ride through every sampling step, so keep it short.",
            inputs=[
                io.Vae.Input("audio_vae", tooltip="The audio VAE (minimax_h3_audio_vae), not the video one."),
                io.Audio.Input("audio"),
                io.Float.Input("max_seconds", default=0.0, min=0.0, max=60.0, step=0.1,
                               tooltip="Trim the reference to this many seconds. 0 = use it whole. Trimming is the cheapest speedup here."),
                MiniMaxRefs.Input("refs", optional=True, tooltip="Chain input, to stack several references."),
            ],
            outputs=[MiniMaxRefs.Output(display_name="refs")],
        )

    @classmethod
    def execute(cls, audio_vae, audio, max_seconds, refs=None) -> io.NodeOutput:
        waveform = audio["waveform"]  # [B, C, L]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        if max_seconds > 0:
            waveform = waveform[..., :int(max_seconds * vae_sr)]
        z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]

        block = {
            "kind": "audio",
            "ref_audio_t": z.shape[-1],
            "audio_latent": z,
        }
        return io.NodeOutput(list(refs or []) + [block])


class MiniMaxH3ApplyRefs(io.ComfyNode):
    """Attach reference blocks to the conditioning."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ApplyRefs",
            display_name="MiniMax H3 Apply Refs",
            category="model/conditioning/minimax",
            description="Injects reference blocks as DiT conditions. Goes after the conditioning node (stock MiniMax H3 Image to Video, or the Reference to Video one). Combining refs with keyframes also needs MiniMax H3 Fix Cond Payload on the model.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                MiniMaxRefs.Input("refs"),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, conditioning, refs) -> io.NodeOutput:
        if not refs:
            return io.NodeOutput(conditioning)
        return io.NodeOutput(node_helpers.conditioning_set_values(conditioning, {"minimax_refs": list(refs)}))


class MiniMaxH3FixCondPayload(io.ComfyNode):
    """Repair cond_video_latents when keyframes and refs are used together.

    comfy/model_base.py builds the payload with two independent branches, and
    the refs branch overwrites the keyframes' cond_video_latents:

        if keyframes: payload["cond_video_latents"] = [kf["latent"] ...]
        if refs:      payload["cond_video_latents"] = [r["latent"] ...]   # clobbers

    The packed layout, however, reserves rows for BOTH (keyframe 'cond'
    segments first, then the refs), and the forward fills them in that order
    from this one list. With an audio-only reference the second branch leaves
    the list empty and the keyframe rows get nothing at all.

    This wrapper rebuilds the list in layout order, right before the forward.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FixCondPayload",
            display_name="MiniMax H3 Fix Cond Payload",
            search_aliases=["keyframes with refs", "minimax payload fix"],
            category="model/patch/minimax",
            description="Required when you combine keyframes with references. Core's payload builder lets the refs overwrite the keyframes' condition latents; this restores both, in the order the packed layout expects. Harmless when only one of the two is in use.",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()

        def fix_payload_wrapper(executor, x, timestep, context, transformer_options,
                                minimax_payload=None, **kwargs):
            if minimax_payload:
                keyframes = minimax_payload.get("keyframes")
                refs = minimax_payload.get("refs")
                if keyframes and refs:
                    minimax_payload["cond_video_latents"] = (
                        [kf["latent"] for kf in keyframes]
                        + [r["latent"] for r in refs if "latent" in r]
                    )
            return executor(x, timestep, context, transformer_options,
                            minimax_payload=minimax_payload, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "minimax_h3_fix_cond_payload", fix_payload_wrapper)
        return io.NodeOutput(m)


class MiniMaxH3RefAudioExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3RefAudio,
            MiniMaxH3ApplyRefs,
            MiniMaxH3FixCondPayload,
        ]


async def comfy_entrypoint() -> MiniMaxH3RefAudioExtension:
    return MiniMaxH3RefAudioExtension()
