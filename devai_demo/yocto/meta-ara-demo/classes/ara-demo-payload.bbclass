# Install one device-verified payload tarball unchanged at its original absolute path.
#
# The V25 demo runs from Python virtual environments (torch, FunASR, MediaPipe)
# that have no Yocto recipes.  Their pyvenv.cfg, shebangs and the release
# config.json hold absolute paths under /root/devai_ara, so each tarball is
# extracted at exactly the path it was harvested from.
#
# A recipe sets:
#   ARA_PAYLOAD      = "<name>"           (tarball <name>.tar in ARA_DEMO_PAYLOAD_DIR)
#   ARA_PAYLOAD_DEST = "/abs/parent/dir"  (directory the tarball's top entry goes into)

LICENSE ?= "CLOSED"

ARA_PAYLOAD ?= ""
ARA_PAYLOAD_DEST ?= ""

python () {
    payload_dir = d.getVar('ARA_DEMO_PAYLOAD_DIR')
    if not payload_dir:
        raise bb.parse.SkipRecipe('ARA_DEMO_PAYLOAD_DIR is not set in local.conf')
    if not d.getVar('ARA_PAYLOAD') or not d.getVar('ARA_PAYLOAD_DEST'):
        bb.fatal('ARA_PAYLOAD and ARA_PAYLOAD_DEST are required')
}

SRC_URI = "file://${ARA_DEMO_PAYLOAD_DIR}/${ARA_PAYLOAD}.tar;unpack=0 \
           file://${ARA_DEMO_PAYLOAD_DIR}/payload_manifest.json"

S = "${UNPACKDIR}"

# Prebuilt aarch64 binaries from the device: keep them byte-identical.
INHIBIT_PACKAGE_STRIP = "1"
INHIBIT_PACKAGE_DEBUG_SPLIT = "1"
INHIBIT_SYSROOT_STRIP = "1"
EXCLUDE_FROM_SHLIBS = "1"
INSANE_SKIP:${PN} += "already-stripped ldflags file-rdeps dev-so staticdev arch textrel libdir build-deps buildpaths rpaths"

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    # Where the fetcher places an absolute file:// path differs between releases.
    tarball=$(find ${UNPACKDIR} -name "${ARA_PAYLOAD}.tar" | head -n 1)
    manifest=$(find ${UNPACKDIR} -name payload_manifest.json | head -n 1)
    [ -f "$tarball" ] && [ -f "$manifest" ] || bbfatal "payload ${ARA_PAYLOAD}.tar or payload_manifest.json not unpacked"
    expected=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['payloads'][sys.argv[2]]['sha256'])" "$manifest" "${ARA_PAYLOAD}")
    actual=$(sha256sum "$tarball" | cut -d' ' -f1)
    if [ "$expected" != "$actual" ]; then
        bbfatal "${ARA_PAYLOAD}.tar sha256 $actual does not match payload_manifest.json ($expected)"
    fi
    install -d ${D}${ARA_PAYLOAD_DEST}
    tar -C ${D}${ARA_PAYLOAD_DEST} --no-same-owner --numeric-owner -xf "$tarball"
}

FILES:${PN} = "${ARA_PAYLOAD_DEST}"
