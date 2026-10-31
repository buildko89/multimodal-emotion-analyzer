SUMMARY = "Resident ARA-2 LLM HTTP service (127.0.0.1:18081) with its Python 3.14 venv and systemd unit"
DESCRIPTION = "Payload harvested from /root/devai_ara/ara06_minimal_llm_20261013_v1 (venv, lock files) \
and /root/devai_ara/services/ara_llm (modules).  The unit file is identical to the device's \
/etc/systemd/system/ara-llm.service (sha256 19b91f6f...)."

inherit ara-demo-payload systemd

ARA_PAYLOAD = "ara-llm-runtime"
ARA_PAYLOAD_DEST = "/root/devai_ara"

SRC_URI += "file://ara-llm.service"

SYSTEMD_SERVICE:${PN} = "ara-llm.service"
SYSTEMD_AUTO_ENABLE = "enable"

# The venv's home is /usr/bin (python3.14 from the BSP); the ARA-2 runtime comes from imx-nxp-ara2.
RDEPENDS:${PN} += "python3 imx-nxp-ara2 ara-llm-qwen25-7b-instruct"

do_install:append() {
    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${UNPACKDIR}/ara-llm.service ${D}${systemd_system_unitdir}/ara-llm.service
}

FILES:${PN} += "${systemd_system_unitdir}/ara-llm.service"
