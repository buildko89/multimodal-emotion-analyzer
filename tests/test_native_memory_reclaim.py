"""Allocator behavior is mocked: test policy and failure isolation, not glibc."""
import unittest
from unittest.mock import patch
from modules import native_memory_reclaim as memory


class NativeMemoryTests(unittest.TestCase):
    def test_native_only_does_not_collect_python_objects(self):
        calls=[]
        with patch.object(memory,'_load_api',return_value=(lambda pad:calls.append(pad) or 1,'mock')),patch.object(memory.gc,'collect',side_effect=AssertionError('GC must be explicit')):
            result=memory.release_idle_pages()
        self.assertEqual(calls,[0]);self.assertEqual(result['status'],'released');self.assertIsNone(result['python_collected'])

    def test_explicit_collection_precedes_trim(self):
        calls=[]
        with patch.object(memory,'_load_api',return_value=(lambda pad:calls.append('trim') or 1,'mock')),patch.object(memory.gc,'collect',side_effect=lambda:calls.append('gc') or 5):
            result=memory.release_idle_pages(collect_cycles=True)
        self.assertEqual(calls,['gc','trim']);self.assertEqual(result['python_collected'],5)

    def test_no_free_pages_is_a_valid_result(self):
        with patch.object(memory,'_load_api',return_value=(lambda pad:0,'mock')):
            self.assertEqual(memory.release_idle_pages()['status'],'no_free_pages')

    def test_unsupported_runtime_does_not_collect_or_raise(self):
        with patch.object(memory,'_load_api',return_value=(None,None)),patch.object(memory.gc,'collect',side_effect=AssertionError('not supported')):
            self.assertEqual(memory.release_idle_pages(collect_cycles=True)['status'],'unavailable')

    def test_api_failure_is_reported_without_destroying_output(self):
        def fail(pad):raise OSError('probe')
        with patch.object(memory,'_load_api',return_value=(fail,'mock')):
            result=memory.release_idle_pages()
        self.assertEqual(result['status'],'error');self.assertIn('probe',result['reason'])

    def test_non_linux_does_not_load_a_library(self):
        memory._load_api.cache_clear()
        try:
            with patch.object(memory.sys,'platform','win32'),patch.object(memory.ctypes,'CDLL',side_effect=AssertionError('must not load')):
                self.assertEqual(memory._load_api(),(None,None))
        finally:memory._load_api.cache_clear()


if __name__=='__main__':unittest.main()
