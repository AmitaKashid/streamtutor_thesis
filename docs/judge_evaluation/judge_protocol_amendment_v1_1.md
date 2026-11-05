\# Judge Model Selection Protocol Amendment v1.1



\## Status



This amendment was frozen before acquisition, loading, smoke testing or quality evaluation of Prometheus 7B v2.0, CompassJudger-2-7B-Instruct or FARE-20B.



Protocol v1 remains preserved unchanged. Protocol v1.1 consists of `judge\_protocol\_v1.md` together with this amendment. If this amendment conflicts with protocol v1, this amendment controls only the explicitly listed operational details.



The research questions, evaluation data design, rubric, schema, admission thresholds, bias tests and lexicographic model-selection rule are unchanged.



\## Reason for amendment



Protocol v1 correctly identifies the NVIDIA A30 24 GB as the deployment target, but its incumbent-reference subsection describes an earlier A100 deployment condition. The executed baseline and all candidate comparisons use the A30.



Protocol v1 also requires generation parameters and quantization to be recorded, but does not freeze their values. This amendment removes that ambiguity before candidate results are observed.



\## Comparison hardware



All admission and comparative experiments use:



\- GPU: NVIDIA A30, 24 GB

\- one judge server per GPU allocation

\- no concurrent workload on the judge GPU

\- warm-up requests excluded from reported latency

\- identical Slurm resource requests where technically possible



The GLM-4-32B-0414 Q4\_K\_M incumbent is treated as the A30 baseline. Earlier A100 execution is historical context and is not used for candidate selection.



\## Candidate deployment artifacts



The source checkpoints remain:



\- `prometheus-eval/prometheus-7b-v2.0`

\- `opencompass/CompassJudger-2-7B-Instruct`

\- `Salesforce/FARE-20B`



Deployment uses Q4\_K\_M GGUF artifacts derived from those checkpoints.



For every artifact, record:



\- source checkpoint

\- GGUF repository

\- exact filename

\- repository revision or commit when available

\- file size

\- SHA-256 checksum

\- download timestamp

\- chat-template source



A third-party quantized artifact is not treated as an independent model. It is treated as a deployment representation of the frozen source checkpoint.



\## Frozen inference configuration



All candidates use:



\- inference backend: `llama-cpp-python 0.3.20`

\- context length: 4096 tokens

\- maximum generated tokens: 360

\- temperature: 0.2

\- top-p: 0.9

\- seed: 0

\- CPU threads: 12

\- GPU layers: -1

\- batch size: 512

\- micro-batch size: 512

\- flash attention: enabled

\- K/Q/V offload: enabled

\- one warm-up request before measured requests



The candidate registry keys must not trigger model-family-specific sampling overrides in `remote\_llm\_server.py`.



\## Prompt compatibility



Each specialist candidate receives two compatibility checks:



1\. Native-format check: the model must execute its documented evaluation prompt format and produce a parseable native judgment.

2\. Application-format check: the model must execute the frozen StreamTutor judge prompt and produce schema-valid JSON.



The native-format check establishes that the deployed artifact retains its intended evaluator behavior. The application-format check establishes compatibility with StreamTutor.



The frozen StreamTutor rubric, schema and retry policy are not adapted separately for individual candidates after results are observed.



\## FARE two-hour admission window



The FARE compatibility window begins immediately before the first acquisition command is executed.



The window includes:



\- artifact acquisition

\- checksum generation

\- registry configuration

\- model loading

\- native-format compatibility generation

\- application-format compatibility generation



Slurm queue waiting time is recorded separately and excluded because it measures cluster availability rather than model compatibility.



FARE passes the compatibility stage only if, within two hours of active acquisition and testing time, it:



1\. is acquired successfully;

2\. loads on one A30;

3\. executes its documented prompt format;

4\. produces a parseable native judgment; and

5\. produces an output that can enter the frozen StreamTutor validation path.



Failure is retained as an experimental result. FARE is not replaced after candidate benchmark results are observed.



\## Version control



This amendment must be committed before the FARE acquisition timer begins. The Git commit, Slurm job ID, model artifact checksum and runtime versions are recorded with the experiment.

