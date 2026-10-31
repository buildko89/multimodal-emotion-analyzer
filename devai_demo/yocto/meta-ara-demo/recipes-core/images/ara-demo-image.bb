SUMMARY = "imx-image-full plus the device-verified V25 ARA demo (FRDM-IMX95 + Kinara ARA-2)"
DESCRIPTION = "Same BSP image the demo was verified on (LF6.18.20-2.0.0, fsl-imx-xwayland), \
with the V25 release, its Python runtimes, the ARA-2 LLM service and model."

require recipes-fsl/images/imx-image-full.bb

IMAGE_INSTALL:append = " packagegroup-ara-demo"

# The verified device had only about 215 MB free on a 29.9 GB root file system,
# which blocked new models (e.g. an ONNX SenseVoice).  Reserve 4 GiB of headroom (in KiB).
IMAGE_ROOTFS_EXTRA_SPACE = "4194304"
