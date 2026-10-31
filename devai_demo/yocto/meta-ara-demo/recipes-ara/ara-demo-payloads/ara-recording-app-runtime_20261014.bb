SUMMARY = "Device-verified app runtime: CPython 3.12.10, app venv (MediaPipe/OpenCV), SenseVoice/Emotion2Vec venv and models, face/hand landmarker models"
DESCRIPTION = "Harvested unchanged from /root/devai_ara/ara06_recording_app_20261014_v1 on the FRDM-IMX95 that passed perf_live12 (2026-10-31)."

inherit ara-demo-payload

ARA_PAYLOAD = "ara06-recording-app"
ARA_PAYLOAD_DEST = "/root/devai_ara"

# The standalone CPython and the venvs link against the BSP's glibc/libstdc++ and use ALSA/V4L2.
RDEPENDS:${PN} += "bash glibc libstdc++ alsa-lib"
