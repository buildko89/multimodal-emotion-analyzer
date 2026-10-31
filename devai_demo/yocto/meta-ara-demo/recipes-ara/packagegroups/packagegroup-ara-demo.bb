SUMMARY = "Everything the V25 ARA demo needs on top of imx-image-full"

inherit packagegroup

RDEPENDS:${PN} = "\
    ara-demo-release \
    ara-recording-app-runtime \
    ara-llm-service \
    ara-llm-qwen25-7b-instruct \
    imx-nxp-ara2 \
    tensorflow-lite \
    tensorflow-lite-neutron-delegate \
    litert-neutron-delegate \
    onnxruntime \
    neutron \
    python3 \
    python3-numpy \
    alsa-utils-aplay \
"
