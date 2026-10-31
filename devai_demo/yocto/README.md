# devai_demo/yocto

`meta-ara-demo` packages the device-verified V25 demo (deployed commit `6723061`) into an NXP i.MX BSP
image (LF6.18.20-2.0.0, `imx-image-full`, wrynose) for the FRDM-IMX95 + Kinara ARA-2.

- `meta-ara-demo/` — the Yocto layer (recipes only; not yet built with bitbake).
- `tools/harvest_device_payload.py` — streams the verified directories from the device over SSH into
  `<out>/*.tar` and `payload_manifest.json`. Those tarballs are proprietary/large and are **never committed**.

Full procedure, design decisions and open items: `ara_spec/04_YOCTO_INTEGRATION.md` (outside this repository,
in `D:\source\repos\ara\ara_spec`).
