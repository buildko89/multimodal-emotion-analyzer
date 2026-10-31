SUMMARY = "Qwen2.5-7B-Instruct compiled for ARA-2 (model.dvm, Ara240 2.0.4) and its tokenizer"
DESCRIPTION = "Harvested unchanged from /usr/share/llm/Qwen2.5-7B-Instruct (about 7.7 GB). \
LICENSE, README.md and SBOM-Qwen2.5-7B-Instruct-Ara240-2.0.4.spdx.json are kept inside the payload. \
Redistribution terms must be confirmed before shipping an image outside the team."

inherit ara-demo-payload

ARA_PAYLOAD = "qwen25-7b-ara240"
ARA_PAYLOAD_DEST = "/usr/share/llm"
