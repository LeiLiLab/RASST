# 决策日志

每条一行：日期（PT）、问题、选了什么、为什么、怎么撤回。

- 2026-11-18｜维基合成 retriever 数据的许可｜HF 数据集标 CC BY-NC 4.0｜语音是用 GigaSpeech 说话人音频做 prompt 合成的，GigaSpeech 音频只许非商业研究与教学使用；句子由 Gemini 2.5 Flash 生成｜改 `code/rasst/manifests/retriever_wiki_synth_dataset.json` 的 `license`，重跑 `prepare` 后用 `upload --refresh-text` 只更新数据卡
- 2026-11-18｜维基合成数据发布哪些行｜289 万行全发，用 `in_hn1024_train` 标出 hn1024 实际用过的 2,567,645 行（trainer 的 `--wiki_rank 1000000` 过滤掉了 326,060 行）｜全量多给数据，又能按列精确复现 hn1024 的训练集｜下载后按 `in_hn1024_train` 过滤即可，不需要重传
- 2026-11-18｜维基合成数据的音频发哪一份｜只发训练用的切片（2.88 至 5.76 秒，FLAC 无损重编码，约 220 GB），不发整句 TTS 音频和 MFA TextGrid｜切片就是 trainer 读的输入，整句音频再加几百 GB，对复现没有增量｜整句音频在 gemini 盘 `wiki_synth_tts_3variant_gigaspeech_full*/` 和同事回传的目录，需要时另开 config 追加上传
- 2026-11-18｜发布版 `utter_id`｜用句子表的行号（`wiki_synth_NNNNNNN`），训练行按 TTS wav 路径对到句子｜trainer 内部 id 在第二批 TTS 上整体偏移 93,709，直接用会让训练行和句子表对不上｜无需撤回；需要 trainer 原 id 时可从源 JSONL 按路径反查
- 2026-11-18｜SFT 切片边界里的 chunk multiplier 用哪个｜JSONL 的 `merge_multiplier` 能复现磁盘切片就用它，否则用从第一个切片长度反推的值，并逐个核对全部切片长度｜de、ja 的 `merge_multiplier` 在一万一千多行上写成 1，和训练实际用的切片不符，模型是按切片训练的｜改 `code/rasst/tools/hf_release_slm_dataset.py` 的 `build_clip_map` 后重跑 `prepare` 和 `upload`
- 2026-11-18｜已发布的 de、ja SFT JSONL 里错误的 `merge_multiplier` 是否改掉｜不改 JSONL，在数据卡和 `audio_clip_map.parquet` 里说明并给出正确值｜JSONL 是模型实际训练用的文件，改动会让发布数据和训练输入不一致｜在 `stage_jsonl` 里按 clip map 改写该字段后重新上传
- 2026-11-18｜`aligned_utterances.tsv` 带哪些列｜只带边界、trajectory 长度、multiplier 和我们生成的目标语言译文，不带英文转写（`src_text`、`asr`、`src_trajectory`）｜GigaSpeech 条款把转写算作数据库的一部分，只许非商业使用且转给他人前需对方同意条款｜在 `load_alignment` 的 `ALIGNMENT_TEXT_COLUMNS` 里加列后重跑
