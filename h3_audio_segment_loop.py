# -*- coding: utf-8 -*-
"""
H3 Audio-Segment Loop Runner

Drives the Herrgotts H3-Infinite start/continue chain from one long song, one
segment at a time. Each segment's own song slice is encoded into that segment's
target audio latent and locked with an all-zero audio mask, so H3 denoises video
only and the avatar lip-syncs to the real song instead of inventing its own
audio track.

Clips 2+ protect their 39-frame video head and the handover always ends on the
previous clip's last frame, so nothing has to be trimmed away: the music cursor
advances by exactly the frames that reach the output and the stitched video stays
in sync with the song from beginning to end.

Everything is written straight into an mp4 while it is generated (fp16 frames),
so a five-minute track costs one segment of memory instead of the whole frame
sequence. Pair it with H3AudioSegVideoOutput to choose where the file lands and
to preview/play it.
"""
import os
import shutil
import sys
import importlib
import time
from fractions import Fraction

import av
import numpy as np
import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.samplers
import comfy.model_management as mm
import folder_paths
from comfy_api.latest import io as api_io, ui as api_ui

FPS = 24.0
AUDIO_HZ = 40.0  # H3 audio latent ticks per second (comfy.ldm.minimax.model.FRAME_RESCALE is 40/24)
STREAM_CRF = 18


class _Mp4Writer:
    """Stream fp16 frames and audio into one H.264/AAC mp4 at constant memory.

    The file starts out in ComfyUI's temp folder as `<name>.mp4.part` and is renamed on close, so a
    run that dies halfway leaves an obviously unfinished file instead of an unplayable mp4.
    H3AudioSegVideoOutput moves the finished file into the output folder.
    """

    def __init__(self, fps, crf):
        self.fps = float(fps)
        self.crf = int(crf)
        self.path = None
        self.container = None
        self.vstream = None
        self.astream = None
        self.sr = None
        self.layout = None
        self.frames = 0
        self.samples = 0

    def _open(self, images, sample_rate, channels):
        h, w = int(images.shape[1]), int(images.shape[2])
        if w % 2 or h % 2:
            raise ValueError(f"streaming output needs even dimensions, got {w}x{h}")
        if channels not in (1, 2):
            raise ValueError(f"streaming output supports mono/stereo audio, got {channels} channels")
        # Streamed into the temp folder; H3AudioSegVideoOutput moves it where the user wants it.
        folder = folder_paths.get_temp_directory()
        self.path = os.path.join(folder, f"h3_audioseg_{int(time.time())}_{os.getpid()}.mp4")
        counter = 0
        while os.path.exists(self.path) or os.path.exists(self.path + ".part"):
            counter += 1
            self.path = os.path.join(
                folder, f"h3_audioseg_{int(time.time())}_{os.getpid()}_{counter}.mp4")
        self.sr = int(sample_rate)
        self.layout = "mono" if channels == 1 else "stereo"
        self.container = av.open(self.path + ".part", mode="w", format="mp4",
                                 options={"movflags": "use_metadata_tags+faststart"})
        self.vstream = self.container.add_stream("h264", rate=Fraction(int(self.fps), 1))
        self.vstream.codec_context.max_b_frames = 0
        self.vstream.codec_context.time_base = Fraction(1, int(self.fps))
        self.vstream.width = w
        self.vstream.height = h
        self.vstream.pix_fmt = "yuv420p"
        self.vstream.options = {"crf": str(self.crf)}
        self.astream = self.container.add_stream("aac", rate=self.sr, layout=self.layout)

    def write(self, images, audio):
        if self.container is None:
            self._open(images, audio["sample_rate"], int(audio["waveform"].shape[1]))
        for frame in images:
            img = (frame * 255.0).clamp(0, 255).byte().detach().cpu().numpy()
            out = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(format="yuv420p")
            out.pts = self.frames
            out.time_base = Fraction(1, int(self.fps))
            for packet in self.vstream.encode(out):
                self.container.mux(packet)
            self.frames += 1
        wave = audio["waveform"]
        if int(wave.shape[-1]) > 0:
            arr = wave[0].float().detach().cpu().contiguous().numpy()
            out = av.AudioFrame.from_ndarray(np.ascontiguousarray(arr), format="fltp", layout=self.layout)
            out.sample_rate = self.sr
            out.pts = self.samples
            out.time_base = Fraction(1, self.sr)
            for packet in self.astream.encode(out):
                self.container.mux(packet)
            self.samples += int(wave.shape[-1])

    def close(self):
        if self.container is None:
            return
        for packet in self.vstream.encode(None):
            self.container.mux(packet)
        for packet in self.astream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.container = None
        os.replace(self.path + ".part", self.path)


def _encode_audio(audio_vae, audio):
    import torchaudio
    waveform = audio["waveform"]
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


def _slice_audio(audio, start_sec, end_sec):
    """Return the audio covering [start_sec, end_sec); callers keep the window non-empty."""
    wav = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    total = wav.shape[-1]
    s = max(0, min(int(round(start_sec * sr)), total))
    e = max(0, min(int(round(end_sec * sr)), total))
    return {"waveform": wav[..., s:e].contiguous(), "sample_rate": sr}


class H3AudioSegmentLoopRunner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "audio": ("AUDIO", {
                    "tooltip": "The song that drives the avatar. It is locked onto every segment's target audio timeline, so the video comes out cut to this song.",
                }),
                "segment_seconds": ("FLOAT", {
                    "default": 6.0, "min": 2.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Song consumed per H3 segment. Longer segments lose less to H3's 17k+5 frame grid and need fewer seams.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scheduler": (_prefer("simple"),),
                "sampler_name": (_prefer_sampler("res_multistep"),),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "first_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("audio", "exact_audio", "video_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "H3 Loop"
    DESCRIPTION = (
        "Slices one song into segments and drives the H3 Infinite start/continue chain with it. Each "
        "segment's target audio latent holds that slice of the song and is locked (audio mask 0), so H3 "
        "denoises only video and the avatar lip-syncs to the real song. Frames are decoded as fp16 and "
        "encoded straight into an mp4 while they are generated, so memory stays at one segment. Connect "
        "video_path to H3AudioSegVideoOutput to choose where the file is kept and to preview/play it. "
        "exact_audio is the same song bit-exact, cut to the video's length, for lossless muxing."
    )

    def _herrgotts(self):
        mod = sys.modules.get("custom_nodes.Herrgotts-H3-Infinite-Continuation-Suite.nodes")
        if mod is None:
            mod = importlib.import_module("custom_nodes.Herrgotts-H3-Infinite-Continuation-Suite.nodes")
        return mod

    def run(self, model, clip, video_vae, audio_vae, audio, segment_seconds,
            prompt, seed=0, width=736, height=1280, steps=8, denoise=1.0,
            scheduler="simple", sampler_name="res_multistep",
            reference_image=None, first_frame=None):
        from comfy_extras.nodes_custom_sampler import (
            RandomNoise, BasicGuider, KSamplerSelect, BasicScheduler, SamplerCustomAdvanced,
        )
        herr = self._herrgotts()
        StartV14 = herr.H3ContinuousStartV14
        ContinueV14 = herr.H3ContinuousContinueV14
        Trim = herr.H3ContinuousTrim
        pixel_frames = herr.pixel_frames
        context_steps = herr.CONTEXT_TO_STEPS

        sr = int(audio["sample_rate"])
        total_sec = audio["waveform"].shape[-1] / sr
        seg = float(segment_seconds)
        writer = _Mp4Writer(FPS, STREAM_CRF)
        print(f"[H3AudioSeg] song {total_sec:.2f}s @ {sr}Hz, {seg:.2f}s segments, locked to the "
              f"target audio timeline, fp16 frames streamed to an mp4", flush=True)

        prev_latent = None
        prev_frames = 0
        all_audio = []
        t_start = time.time()
        cursor = 0.0
        index = 0

        while cursor < total_sec - 0.5 / FPS:
            if mm.processing_interrupted():
                break
            request = min(seg, max(0.5, total_sec - cursor))
            clip_seed = int(seed) + index
            elapsed = time.time() - t_start
            avg = elapsed / max(1, index)
            left = max(0.0, (total_sec - cursor) / seg)
            eta_str = f"~{int(avg * left)}s left" if index > 0 else "estimating..."
            t_clip = time.time()

            if index == 0:
                positive, latent, _ = StartV14().build(
                    clip=clip, vae=video_vae, prompt=prompt,
                    width=width, height=height, duration=request, ref_image_size="match",
                    first_frame=first_frame,
                    qwen_reference_1=reference_image,
                )
                head_frames = 0
            else:
                # Clips 2+ always protect a 39-frame video head. Ending the previous clip on its own
                # last frame makes that head a replay of music that already played, so the new slice
                # starts at frame head_frames and nothing has to be trimmed away.
                handover = {"available": True, "frame_count": prev_frames,
                            "handover_end_frame": prev_frames - 1, "landing_tail_frames": 0}
                positive, latent, head_frames, _, _, _ = ContinueV14().build(
                    clip=clip, vae=video_vae, previous_latent=prev_latent, prompt=prompt,
                    width=width, height=height, duration=request,
                    masked_context_frames="39", audio_feather_ticks=0, ref_image_size="match",
                    duration_mode="Net New Content", audio_tail_carryover="Match Video Handover",
                    handover=handover,
                    qwen_reference_1=reference_image,
                )
                head_frames = int(head_frames)

            target_video, target_audio = herr._streams_from_latent(latent)
            total_t = int(target_audio.shape[-1])
            total_frames = int(pixel_frames(target_video.shape[2]))

            # Audio tick 0 sits on target frame 0, which for clips 2+ is part of the protected head.
            music_start = max(0.0, cursor - head_frames / FPS)
            music_latent, _ = _encode_audio(audio_vae, _slice_audio(
                audio, music_start, music_start + total_t / AUDIO_HZ))
            if int(music_latent.shape[-1]) < total_t:
                music_latent = F.pad(music_latent, (0, total_t - int(music_latent.shape[-1])))
            else:
                music_latent = music_latent[..., :total_t]
            # The audio VAE can hand back a different dtype/device than the empty AV latent (e.g. with
            # --fp16-intermediates), and the sampler packs both streams with a single torch.cat, so match
            # the target exactly, the same way the suite casts its copied context.
            music_latent = music_latent.to(device=target_video.device, dtype=target_video.dtype)

            # Same masked-AV convention the suite builds: 1 = generate, 0 = keep. The protected video
            # head stays, the song is kept everywhere (no audio is generated).
            video_mask = torch.ones((1, 1, *target_video.shape[2:]),
                                    device=target_video.device, dtype=torch.float32)
            if head_frames:
                video_mask[:, :, :context_steps[head_frames]] = 0.0
            audio_mask = torch.zeros((1, 1, *music_latent.shape[2:]),
                                     device=music_latent.device, dtype=torch.float32)

            latent = dict(latent)
            latent["samples"] = comfy.nested_tensor.NestedTensor((target_video, music_latent))
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

            print(f"[H3AudioSeg] >>> segment {index+1}: song {music_start:.2f}.."
                  f"{music_start + total_t / AUDIO_HZ:.2f}s -> {total_frames}f "
                  f"(head {head_frames}f)  {eta_str}", flush=True)

            noise = RandomNoise().execute(noise_seed=clip_seed).result[0]
            guider = BasicGuider().execute(model=model, conditioning=positive).result[0]
            sampler = KSamplerSelect().execute(sampler_name=sampler_name).result[0]
            sigmas = BasicScheduler().execute(model=model, scheduler=scheduler,
                                              steps=steps, denoise=denoise).result[0]
            samples = SamplerCustomAdvanced().execute(
                noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent
            ).result[0]

            video_latent, audio_latent = herr._streams_from_latent(samples)
            images = video_vae.decode(video_latent)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            out_sr = int(getattr(audio_vae, "audio_sample_rate_output",
                                 getattr(audio_vae, "audio_sample_rate", 44100)))
            waveform = audio_vae.decode(audio_latent).movedim(-1, 1)

            images, clip_audio = Trim().trim(
                images, head_frames, 0, fps=FPS,
                audio={"waveform": waveform, "sample_rate": out_sr},
            )
            playable = int(images.shape[0])
            print(f"[H3AudioSeg]     segment {index+1} done: {images.shape} = "
                  f"{playable/FPS:.2f}s, took {int(time.time()-t_clip)}s", flush=True)
            clip_audio = {"waveform": clip_audio["waveform"].detach().cpu(), "sample_rate": out_sr}
            # fp16 halves the segment's footprint and is still exact enough for 8-bit output; the fp32
            # decode is released here because a trimmed clip is only a view into it.
            images = images.half()
            writer.write(images, clip_audio)
            all_audio.append(clip_audio)

            prev_latent = samples
            prev_frames = total_frames
            cursor += playable / FPS
            index += 1
            del images, noise, guider, sampler, sigmas, positive
            mm.soft_empty_cache()

        mm.throw_exception_if_processing_interrupted()
        if not all_audio:
            raise RuntimeError("No segment completed.")

        writer.close()
        done_frames = writer.frames
        video_path = writer.path

        out_sr = all_audio[0]["sample_rate"]
        audio_wave = torch.cat([a["waveform"] for a in all_audio], dim=-1)
        want_out = int(round(done_frames / FPS * out_sr))
        if audio_wave.shape[-1] < want_out:
            audio_wave = torch.cat((audio_wave, torch.zeros(
                (*audio_wave.shape[:-1], want_out - audio_wave.shape[-1]), dtype=audio_wave.dtype)), dim=-1)
        else:
            audio_wave = audio_wave[..., :want_out]
        if audio_wave.ndim == 2:
            audio_wave = audio_wave.unsqueeze(0)
        out_audio = {"waveform": audio_wave, "sample_rate": out_sr}

        # The same song, untouched, cut to the video's length: mux this for a lossless audio track.
        want_src = int(round(done_frames / FPS * sr))
        song = audio["waveform"][:1]
        if song.shape[-1] > want_src:
            song = song[..., :want_src].clone()
        elif song.shape[-1] < want_src:
            song = F.pad(song, (0, want_src - song.shape[-1]))
        exact_audio = {"waveform": song, "sample_rate": sr}

        print(f"[H3AudioSeg] === DONE: {done_frames} frames = {done_frames/FPS:.2f}s of "
              f"{total_sec:.2f}s song, wall {int(time.time()-t_start)}s ===", flush=True)
        print(f"[H3AudioSeg] mp4: {video_path} (H.264 crf {STREAM_CRF} + AAC at {int(FPS)}fps; "
              f"connect H3AudioSegVideoOutput to keep and preview it)", flush=True)
        return (out_audio, exact_audio, video_path)


class H3AudioSegVideoOutput:
    """Move a streamed H3 audio-segment mp4 into the output folder and preview it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "video_path from H3 Infinite - Audio Segment Loop Runner (a finished mp4 in ComfyUI/temp).",
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/H3AudioSeg",
                    "tooltip": "Where to keep the video, relative to ComfyUI's output folder. Subfolders are allowed, e.g. mv/song1.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3 Loop"
    DESCRIPTION = (
        "Moves the video that H3 Infinite - Audio Segment Loop Runner streamed into the temp folder to "
        "the output folder and shows it in the UI, so the finished clip can be watched and downloaded "
        "without ever loading a frame into memory. Returns the final absolute path."
    )

    def save(self, video_path, filename_prefix):
        source = os.path.normcase(os.path.realpath(str(video_path)))
        roots = tuple(os.path.normcase(os.path.realpath(path)) for path in (
            folder_paths.get_temp_directory(), folder_paths.get_output_directory()))
        if not os.path.isfile(source) or not any(source.startswith(root + os.sep) for root in roots):
            raise ValueError(
                "H3AudioSegVideoOutput needs the finished mp4 that the Audio Segment Loop Runner left in "
                f"{roots[0]}, but {video_path!r} is not there. If an earlier run already moved it, queue the "
                "runner again (its seed changes per queue) and reconnect video_path."
            )

        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, f"{name}_{int(counter):05d}_.mp4")
        while os.path.exists(target):
            counter += 1
            target = os.path.join(folder, f"{name}_{int(counter):05d}_.mp4")

        try:
            os.replace(source, target)
        except OSError:
            shutil.move(source, target)  # temp and output on different volumes
        print(f"[H3AudioSeg] kept {target}", flush=True)
        preview = api_ui.PreviewVideo(
            [api_ui.SavedResult(os.path.basename(target), subfolder, api_io.FolderType.output)]
        ).as_dict()
        return {"ui": preview, "result": (target,)}


NODE_CLASS_MAPPINGS = {
    "H3AudioSegmentLoopRunner": H3AudioSegmentLoopRunner,
    "H3AudioSegVideoOutput": H3AudioSegVideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AudioSegmentLoopRunner": "H3 Infinite - Audio Segment Loop Runner",
    "H3AudioSegVideoOutput": "H3 Infinite - AudioSeg Video Output (save + preview)",
}
