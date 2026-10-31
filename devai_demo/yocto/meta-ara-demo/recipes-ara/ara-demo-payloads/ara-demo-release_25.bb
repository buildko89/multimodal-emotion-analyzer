SUMMARY = "ARA V25 demo release (live pipeline, supervisor runtime.py, assets, saved preparation inputs)"
DESCRIPTION = "Harvested unchanged from /root/devai_ara/ara_demo_20261026_v25/release. \
Built from GitHub buildko89/multimodal-emotion-analyzer devai_demo/device_app at commit 6723061 \
by operator/ara_demo.py build/deploy; release/manifest.json holds the sha256 of every file."

inherit ara-demo-payload

ARA_PAYLOAD = "ara-demo-release-v25"
ARA_PAYLOAD_DEST = "/root/devai_ara/ara_demo_20261026_v25"

# runtime.py and the NPU services run on the system Python with the BSP's TFLite / Neutron stack.
RDEPENDS:${PN} += "ara-recording-app-runtime python3 python3-numpy bash alsa-utils-aplay \
                   tensorflow-lite tensorflow-lite-neutron-delegate neutron"

do_install:append() {
    # Session evidence is created at run time; ship the release without any.
    install -d ${D}${ARA_PAYLOAD_DEST}/release/workspace/runs
    if [ -n "$(ls -A ${D}${ARA_PAYLOAD_DEST}/release/workspace/runs)" ]; then
        bbfatal "release payload unexpectedly contains run evidence"
    fi
}
