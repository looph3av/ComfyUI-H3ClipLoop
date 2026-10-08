# -*- coding: utf-8 -*-
"""
H3 Clip Loop Runner + H3 Clip Prompts
"""
import sys
import importlib
import time

import comfy.samplers
import comfy.model_management as mm
import node_helpers

FPS = 24.0


def _encode_ref_audio(audio_vae, audio):
    """Encode a standalone reference audio into an H3 audio latent.

    Mirrors comfy_extras.nodes_minimax_h3._encode_ref_audio: resamples to the
    audio VAE's native rate, then encodes the first channel group as
    [1, 32, 2, T]. Returns (audio_latent, ref_audio_t).
    """
    import torchaudio

    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _prefer(*names):
    return [n for n in names if n in comfy.samplers.SCHEDULER_NAMES] + \
           [n for n in comfy.samplers.SCHEDULER_NAMES if n not in names]


def _prefer_sampler(*names):
    return [n for n in names if n in comfy.samplers.SAMPLER_NAMES] + \
           [n for n in comfy.samplers.SAMPLER_NAMES if n not in names]


class H3ClipPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_2": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_3": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_4": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_5": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_6": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_7": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_8": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_9": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_10": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "build"
    CATEGORY = "H3 Loop"

    def build(self, prompt_1, prompt_2, prompt_3, prompt_4, prompt_5,
              prompt_6, prompt_7, prompt_8, prompt_9, prompt_10):
        filled = [str(p) for p in (prompt_1, prompt_2, prompt_3, prompt_4, prompt_5,
                                   prompt_6, prompt_7, prompt_8, prompt_9, prompt_10)
                  if str(p).strip()]
        print(f"[H3ClipPrompts] collected {len(filled)} non-empty prompts", flush=True)
        return (filled,)


class H3ClipLoopRunner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "clip_count": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "duration": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 60.0, "step": 0.5}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scheduler": (_prefer("simple"),),
                "sampler_name": (_prefer_sampler("res_multistep"),),
                "handover_preset": (["Balanced", "Motion Safe"], {
                    "default": "Balanced",
                    "tooltip": "Freeze/handover detector preset used to find each clip's safe tail cutoff. Motion Safe uses a 6-frame safety margin instead of 3.",
                }),
            },
            "optional": {
                "prompts": ("LIST",),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional Qwen-only reference image. It becomes a Qwen <Picture N> in the presentation (not a native minimax_refs latent).",
                }),
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Optional second Qwen-only reference image. Follows reference_image in <Picture N> order (no first/last frames: Picture 2; with first/last frames connected: shifted after them).",
                }),
                "reference_image_3": ("IMAGE", {
                    "tooltip": "Optional third Qwen-only reference image. Follows reference_image_2 in <Picture N> order.",
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": "Optional standalone reference audio. Encoded with audio_vae and injected into minimax_refs (kind 'audio') for every clip.",
                }),
                "first_frame": ("IMAGE", {
                    "tooltip": "Optional start frame of the chain. Clip 1 uses it as its first temporal keyframe and first Qwen <Picture N>.",
                }),
                "last_frame": ("IMAGE", {
                    "tooltip": "Optional end frame of each segment. Every clip uses it as its last temporal keyframe, so the next clip continues from that endpoint.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "H3 Loop"

    def _herrgotts(self):
        mod = sys.modules.get("custom_nodes.Herrgotts-H3-Infinite-Continuation-Suite.nodes")
        if mod is None:
            mod = importlib.import_module("custom_nodes.Herrgotts-H3-Infinite-Continuation-Suite.nodes")
        return mod

    def run(self, model, clip, video_vae, audio_vae, clip_count,
            seed=0, width=736, height=1280, duration=6.0, steps=8, denoise=1.0,
            scheduler="simple", sampler_name="res_multistep",
            handover_preset="Balanced",
            prompts=None,
            reference_image=None, reference_image_2=None, reference_image_3=None,
            reference_audio=None,
            first_frame=None, last_frame=None):
        import torch
        from comfy_extras.nodes_custom_sampler import (
            RandomNoise, BasicGuider, KSamplerSelect, BasicScheduler, SamplerCustomAdvanced,
        )
        herr = self._herrgotts()
        StartV14 = herr.H3ContinuousStartV14
        ContinueV14 = herr.H3ContinuousContinueV14
        Analyze = herr.H3ContinuousAnalyzeHandoverV14
        Trim = herr.H3ContinuousTrim

        if prompts is None:
            prompts = []
        filled = [str(p).strip() for p in prompts if str(p).strip()]
        if not filled:
            raise ValueError("No prompts: wire H3ClipPrompts into prompts input.")
        if clip_count <= len(filled):
            prompts = filled[:clip_count]
        else:
            prompts = filled + [filled[-1]] * (clip_count - len(filled))
        if len(filled) != clip_count:
            print(f"[H3ClipLoop] {len(filled)} prompts for {clip_count} clips: "
                  f"{'reusing the last prompt' if len(filled) < clip_count else 'extra prompts ignored'}",
                  flush=True)

        ref_audio_block = None
        if reference_audio is not None:
            ref_latent, ref_audio_t = _encode_ref_audio(audio_vae, reference_audio)
            ref_audio_block = [{"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": ref_latent}]
            print(f"[H3ClipLoop] reference audio encoded once: ref_audio_t={ref_audio_t}", flush=True)

        prev_latent = None
        prev_handover = None
        all_images = []
        all_audio = []
        t_start = time.time()

        for i in range(1, clip_count + 1):
            if mm.processing_interrupted():
                break
            clip_seed = int(seed) + i - 1
            elapsed = time.time() - t_start
            avg = elapsed / max(1, i - 1)
            eta = int(avg * (clip_count - (i - 1))) if i > 1 else -1
            eta_str = f"~{eta}s remaining" if eta > 0 else "estimating..."
            print(f"[H3ClipLoop] >>> clip {i}/{clip_count}  ({width}x{height}, {duration}s)  "
                  f"elapsed {int(elapsed)}s  {eta_str}", flush=True)
            t_clip = time.time()

            if i == 1:
                positive, latent, _ = StartV14().build(
                    clip=clip, vae=video_vae, prompt=prompts[0],
                    width=width, height=height, duration=duration, ref_image_size="match",
                    first_frame=first_frame, last_frame=last_frame,
                    qwen_reference_1=reference_image,
                    qwen_reference_2=reference_image_2,
                    qwen_reference_3=reference_image_3,
                )
                head_ctx = 0
            else:
                positive, latent, head_ctx, _, _, _ = ContinueV14().build(
                    clip=clip, vae=video_vae, previous_latent=prev_latent, prompt=prompts[i - 1],
                    width=width, height=height, duration=duration,
                    masked_context_frames="39", audio_feather_ticks=0, ref_image_size="match",
                    duration_mode="Net New Content", audio_tail_carryover="Full Previous Tail",
                    handover=prev_handover, last_frame=last_frame,
                    qwen_reference_1=reference_image,
                    qwen_reference_2=reference_image_2,
                    qwen_reference_3=reference_image_3,
                )

            if ref_audio_block is not None:
                positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": ref_audio_block})

            noise = RandomNoise().execute(noise_seed=clip_seed).result[0]
            guider = BasicGuider().execute(model=model, conditioning=positive).result[0]
            sampler = KSamplerSelect().execute(sampler_name=sampler_name).result[0]
            sigmas = BasicScheduler().execute(model=model, scheduler=scheduler,
                                              steps=steps, denoise=denoise).result[0]
            samples = SamplerCustomAdvanced().execute(
                noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent
            ).result[0]

            # H3 samples is a packed [video, audio] tensor; split before decoding.
            video_latent, audio_latent = herr._streams_from_latent(samples)
            images = video_vae.decode(video_latent)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            # Analyze handover on the FULL decoded images BEFORE trimming (frame_count must
            # match prev_latent which stays complete for the next Continue call).
            handover, hinfo, *_ = Analyze().analyze(images=images, preset=handover_preset)

            sr = int(getattr(audio_vae, "audio_sample_rate_output",
                             getattr(audio_vae, "audio_sample_rate", 44100)))
            waveform = audio_vae.decode(audio_latent).movedim(-1, 1)
            std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            waveform = waveform / std

            # Rendered-output trim with the same rules as the suite's Stitch Ready / Final Clip
            # modes: clips 2+ drop the reused head context, and every clip except the last drops
            # the freeze/brightness tail after its own safe handover boundary. Both streams are
            # trimmed together so the audio samples stay on the video's timeline.
            head_trim = int(head_ctx) if i > 1 else 0
            tail_trim = 0 if i == clip_count else int(handover["landing_tail_frames"])
            images, clip_audio = Trim().trim(
                images, head_trim, tail_trim, fps=FPS,
                audio={"waveform": waveform, "sample_rate": sr},
            )

            clip_secs = images.shape[0] / FPS
            print(f"[H3ClipLoop]     clip {i} done: {images.shape} = {clip_secs:.2f}s  "
                  f"(trim head {head_trim}f / tail {tail_trim}f, took {int(time.time()-t_clip)}s)",
                  flush=True)
            print(f"[H3ClipLoop]     handover: {hinfo}", flush=True)
            # A trimmed clip is a view into the full decoded clip, so holding it would pin the
            # untrimmed remainder in memory. Clone only when a trim actually happened.
            all_images.append(images.clone() if (head_trim or tail_trim) else images)
            all_audio.append({"waveform": clip_audio["waveform"].detach().cpu(), "sample_rate": sr})

            prev_latent = samples
            prev_handover = handover
            del images, noise, guider, sampler, sigmas, positive
            mm.soft_empty_cache()

        mm.throw_exception_if_processing_interrupted()
        if not all_images:
            raise RuntimeError("No clip completed.")

        # Assemble in place and drop each clip as soon as it is copied: torch.cat would hold
        # every clip and the joined result at the same time.
        clips = len(all_images)
        total_frames = sum(int(t.shape[0]) for t in all_images)
        video = torch.empty((total_frames,) + tuple(all_images[0].shape[1:]), dtype=all_images[0].dtype)
        offset = total_frames
        while all_images:
            part = all_images.pop()
            offset -= int(part.shape[0])
            video[offset:offset + int(part.shape[0])].copy_(part)

        total_secs = video.shape[0] / FPS
        total_elapsed = time.time() - t_start
        print(f"[H3ClipLoop] === DONE: {clips}/{clip_count} clips, "
              f"{video.shape[0]} frames = {total_secs:.2f}s, wall {int(total_elapsed)}s ===", flush=True)

        sr = all_audio[0]["sample_rate"]
        audio_wave = torch.cat([a["waveform"] for a in all_audio], dim=-1)
        # Every clip's audio grid is rounded to samples on its own; force the joined track onto
        # the video's exact duration so those roundings cannot accumulate into A/V drift.
        want = int(round(video.shape[0] / FPS * sr))
        if audio_wave.shape[-1] < want:
            audio_wave = torch.cat((audio_wave, torch.zeros(
                (*audio_wave.shape[:-1], want - audio_wave.shape[-1]), dtype=audio_wave.dtype)), dim=-1)
        else:
            audio_wave = audio_wave[..., :want]
        if audio_wave.ndim == 2:
            audio_wave = audio_wave.unsqueeze(0)
        audio = {"waveform": audio_wave, "sample_rate": sr}
        print(f"[H3ClipLoop] audio: {tuple(audio_wave.shape)} @ {sr}Hz", flush=True)

        return (video, audio)


NODE_CLASS_MAPPINGS = {
    "H3ClipLoopRunner": H3ClipLoopRunner,
    "H3ClipPrompts": H3ClipPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ClipLoopRunner": "H3 Infinite - Clip Loop Runner (N-segment loop)",
    "H3ClipPrompts": "H3 Infinite - Clip Prompts (10 prompts)",
}