SUMMARY = "Everything the V25 ARA demo needs on top of imx-image-full"

# The dependencies are machine-specific BSP packages, and some are renamed at
# packaging time (tensorflow-lite -> libtensorflow-lite2.19.0), which an allarch
# packagegroup must not depend on. This has to be set before the inherit.
PACKAGE_ARCH = "${MACHINE_ARCH}"

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
