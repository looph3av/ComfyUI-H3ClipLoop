# ComfyUI-H3ClipLoop

把 MiniMax H3 的连续续写跑成 N 段视频：`Clip Loop Runner` 输出拼接好的整段 IMAGE + AUDIO；`Audio Segment Loop Runner` 用一首歌驱动数字人，边生成边写 mp4（内存恒定），再由 `AudioSeg Video Output` 决定落盘位置并预览播放。

## 依赖

- **Herrgotts-H3-Infinite-Continuation-Suite**（必需）：本节点直接调用它的 `H3ContinuousStartV14` / `H3ContinuousContinueV14` / `H3ContinuousAnalyzeHandoverV14` / `H3ContinuousTrim`，同目录名必须是 `Herrgotts-H3-Infinite-Continuation-Suite`。
- 采样链与套件示例一致：`model` 直接用 UNETLoader 的输出即可（H3 模型自带 shift 12 / audio_shift 3），需要调 shift 时再接 `ModelSamplingMiniMaxH3`（`MiniMaxH3SigmaShift`）；`video_vae` / `audio_vae` 分别是 H3 的 video VAE 和 audio VAE。

## 节点

- **H3 Infinite - Clip Loop Runner**：按 `clip_count` 循环采样，逐段解码、裁剪、最后拼接输出。
- **H3 Infinite - Clip Prompts**：最多 10 条提示词，接到 Runner 的 `prompts` 输入。提示词少于段数时重复最后一条，多余时忽略（两种情况都会打印一行日志）。
- **H3 Infinite - Audio Segment Loop Runner**：用一首歌驱动，把歌切片后**逐段锁进目标音频 latent**（audio mask 全 0），H3 只生成视频、口型跟着真音乐；`cursor` 按实际进入成片的帧数推进，段与段的音乐区间重叠正好 39 帧（第 2 段起被保护的头）。**帧固定 fp16、边生成边写 mp4**，输出 `audio`（锁定后的音轨）/ `exact_audio`（原曲逐样本、裁到视频长度）/ `video_path`（temp 里的成品 mp4）。
- **H3 Infinite - AudioSeg Video Output**：把上面写好的 mp4 移到 `output/<filename_prefix>_00001_.mp4`，并在界面里直接预览/播放**完整视频**，输出最终绝对路径。`filename_prefix` 就是"写盘路径"（相对 ComfyUI 的 output 目录，可带子目录，如 `mv/song1`）。

## 内存与长音频

- Clip Loop Runner 是"整段 IMAGE"节点：帧全部常驻内存，`clip_count` 大、分辨率高时占用很高。需要超长视频时走套件的分段落盘路线：`H3ContinuousSaveLatent` 存每段 latent（每段只有数 MB，比 fp32 解码帧小两个数量级），再用 `H3ContinuousStitchSavedChainV14` 逐段解码直接写 mp4（内存与段数无关）。
- Audio Segment Loop Runner **永远不累积帧**：每段解码后立刻转 fp16 并交给 PyAV 编码进 mp4，内存只有当前一段（5 分钟歌 @480p 从 ~34GB 降到 <1GB），因此它没有 IMAGE 输出，也没有精度/模式开关。
- 中间文件先写成 `ComfyUI/temp/h3_audioseg_*.mp4.part`，结束时改名；中途失败只留下明显未完成的 `.part`。接了 `H3 Infinite - AudioSeg Video Output` 才会移进 output 并在界面里预览/播放；不接就留在 temp（日志里有完整路径）。
- mp4 是 H.264 crf18 + AAC、固定 24fps。AAC 是有损的，要无损音轨请用 `exact_audio` 重新封装：`ffmpeg -i out.mp4 -i song.wav -map 0:v -map 1:a -c:v copy -c:a flac out.mkv`。

## 关键语义

- `first_frame`：整条链的起始帧（只有第 1 段使用）。
- `last_frame`：**每个分段各自的末帧**。每段都把它作为结尾关键帧，下一段从该端点继续，因此所有分段共用同一张末帧图。
- 渲染输出裁剪与套件的 Stitch Ready / Final Clip 规则一致：第 2 段起裁掉复用的头部上下文；除最后一段外，裁掉各自 handover 边界之后的冻结/亮度尾巴（`handover_preset` 决定检测器的安全边距）。视频帧与音频采样点在同一次裁剪里处理，拼接后再把音轨长度对齐到视频时长，避免逐段累积音画漂移。
- 每段裁剪结果以紧凑副本累积（`Trim` 返回的是视图，直接保存会把未裁剪的整段解码结果钉在内存里），收尾用「预分配 + 逐段拷贝并即时释放」而不是 `torch.cat`，避免拼接瞬间的双份峰值。

## 日志

每次运行会打印每段的 `trim head/tail`、handover 状态、累计帧数/秒数以及最终音频采样数，便于核对接缝与音画长度。
