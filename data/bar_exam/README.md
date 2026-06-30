# MOJ Korean Bar Exam Dataset Worklog

Workspace: `/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text`

## Current Priority

선택형(다지선다)을 먼저 완성한다. 기록형, 사례형, 선택과목은 이후 타입별 Hugging Face 데이터셋으로 분리한다.

## Source Pages

- 법무부 시험자료실 기출문제: <https://www.moj.go.kr/moj/405/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGbW9qJTJGMTUwJTJGYXJ0Y2xMaXN0LmRvJTNGdGFibGVfY2F0ZV9zZWxlY3QlM0QxNDElMjZiYnNDbFNlcSUzRDE0MSUyNmlzVmlld01pbmUlM0RmYWxzZSUyNmJic09wZW5XcmRTZXElM0QlMjZzcmNoQ29sdW1uJTNEc2olMjZzcmNoV3JkJTNEJTI2>
- 법무부 변호사시험 공지사항 정답가안/확정정답: <https://www.moj.go.kr/moj/2126/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGbW9qJTJGMTUxJTJGYXJ0Y2xMaXN0LmRvJTNGYmJzQ2xTZXElM0QlMjZpc1ZpZXdNaW5lJTNEZmFsc2UlMjZiYnNPcGVuV3JkU2VxJTNEJTI2c3JjaENvbHVtbiUzRHNqJTI2c3JjaFdyZCUzRCVFQiVCMyU4MCVFRCU5OCVCOCVFQyU4MiVBQyVFQyU4QiU5QyVFRCU5NyU5OCslRUMlQTAlOTUlRUIlOEIlQjUlRUElQjAlODglRUMlOTUlODglMjY%3D>
- 공공누리 제1유형: <https://www.kogl.or.kr/info/licenseType1.do>

## Local Outputs

- 선택형 processed CSV/JSONL: `data/bar_exam/processed_multiple_choice/`
- 제15회 v5 근거 패킷: `data/bar_exam/round15_rag_contexts_v5_20260629/`
- 제15회 v5 만점 풀이: `data/bar_exam/15th_solved_v5/`
- 제15회 원문 분리 JSON: `data/bar_exam/15th_split/`
- 선택형 Hugging Face upload folder/model card: `data/bar_exam/hf_multiple_choice/`
- Raw downloads: `data/bar_exam/raw/downloads/`
- Extracted ZIP contents: `data/bar_exam/extracted/`
- HWP/HWPX extracted text: `data/bar_exam/text/`

## Reproduction

```bash
python3 scripts/build_moj_bar_exam_dataset.py --only-multiple-choice
python3 scripts/build_moj_bar_exam_dataset.py --only-multiple-choice --upload
```

Default HF dataset repo for the first command with upload is:

```text
<hf-username>/korean-bar-exam-moj-multiple-choice
```

## Selection-Type QA

Latest selected/multiple-choice build:

- Source attachments selected: 18
- Source HWP/HWPX documents after ZIP extraction: 66
- Parsed question rows: 2,250
- Parsed answer rows: 2,250
- Multiple-choice count issues: 0
- Rows with full parsed choices and answer: 2,244
- Rows with answer but unusual choice layout requiring review in `choices_json`: 6

All 2,250 rows have `question_text` and `answer`.

## SFT Use in This Repository

The SFT repository does not blindly train every answer value in
`processed_multiple_choice/questions.csv`. Some answer fields contain polluted
or non-normalized values such as `30`, `37`, or `40`, so the SFT builder only
uses answer labels that are already exact `1`~`5`.

Current CPU-only context-solver preparation:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_prepare_bar_exam_v5_sft.sh
```

Prepared output:

```text
/home/work/.data/lfm2_ko_sft/prepared/bar_exam_v5/20260630_bar_exam_v5_context_solver_8192/lfm_chat_8192
```

Final validated size:

| category | rows |
|---|---:|
| 1~14 safe MCQA, `①~⑤` input | 1,888 |
| 1~14 safe MCQA, `1.~5.` input | 1,888 |
| current-law simple | 1,000 |
| current-law hard | 1,000 |
| 15th v5 answer-free procedure | 150 |
| legal search first action | 150 |
| 15th solved-v5 full solution | 150 |
| 15th solved-v5 compact answer | 150 |

After dedupe/tokenization this becomes 6,374 samples and 5,863,863 LFM tokens
at max sequence length 8192.

The purpose is not to memorize only the 15th exam. The actual supervised pair is:

```text
15th_split original problem
+ round15_rag_contexts_v5_20260629 evidence packet
-> 15th_solved_v5 grounded full solution
```

This trains the model to read a supplied legal evidence packet, apply the
manual verification points, judge each option, and output a normalized answer.
See `docs/BAR_EXAM_V5_CONTEXT_SFT_PLAN_20260630.ko.md` for the full rationale.

## License Note

Source exam content is under Korea Open Government License Type 1 (KOGL Type 1). The dataset card uses `license: other` and records KOGL attribution conditions. The local packaging/conversion script may follow the repository Apache-2.0 license, but source content is not relicensed away from KOGL Type 1.

## Next Type Splits

Planned separate outputs/HF repos after selection-type upload:

- `korean-bar-exam-moj-case-type` for 사례형
- `korean-bar-exam-moj-record-type` for 기록형
- `korean-bar-exam-moj-elective-case-type` for 선택과목 사례형
