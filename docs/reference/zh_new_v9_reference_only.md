# Chinese new_v9 SLM Path: Reference Only

This document preserves the original zh handling path used by the submitted
paper-exact table. It is not the release-canonical SLM reproduction recipe.

The release-canonical recipe is:

```text
cap16_denoise_budget_ttag for de, ja, and zh
```

Use:

```bash
cd /mnt/taurus/data2/jiaxuanluo/RASST
bash code/rasst/scripts/reproduce_slm.sh --lang zh --stage all
```

## Reference Provenance

The older zh path used the `new_v9_assistant_termtag_delay_clean_no_gt_zero`
family.

Known legacy launchers:

```text
code/legacy/documents/code/train/sst_omni_train/launchers/2026/05/20260523__build_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh.sh
code/legacy/documents/code/train/sst_omni_train/launchers/2026/05/20260523__speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_taurus8_r32a64_tp2.sh
```

Known legacy train manifest:

```text
code/legacy/documents/code/train/sst_omni_train/manifests/2026/05/20260523T222729__speech_llm_train__new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8.json
```

Known exported HF model path recorded by the submitted-paper exact manifest:

```text
/mnt/gemini/data1/jiaxuanluo/slm_exports/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32/v0-20260524-062743-hf
```

Keep this path available for provenance comparison, but do not present it as the
main release reproduction workflow.
