"""Fail closed on lost capture; reconnect uses a fresh, explicitly owned attempt."""
def capture_fault(shared, role, error, now_ns):
    fault={'role':role,'error':str(error),'available_ns':now_ns}
    shared.setdefault('input_faults',[]).append(fault)
    shared['status']='入力停止：接続を確認してから再開してください'
    return fault


def stale_capture(shared, now_ns, started_ns, *, startup_seconds=15, timeout_seconds=3):
    if not shared.get('sensor_capture_enabled',True) or now_ns-started_ns < startup_seconds*1e9:
        return None
    for role,key in (('camera','camera_native_ns'),('microphone','audio_available_ns')):
        last=shared.get(key)
        if last is None or now_ns-last > timeout_seconds*1e9:
            return role
    return None
