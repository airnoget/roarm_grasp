import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gpu_download import download

TEST_ROOT=Path(__file__).resolve().parent/'.test-artifacts'
TEST_ROOT.mkdir(exist_ok=True)


class Response(io.BytesIO):
    def __init__(self, data, status=200, start=0, fail=False):
        super().__init__(data)
        self.status=status; self.fail=fail; self.calls=0
        self.headers={'Content-Length':str(len(data))}
        if status==206:
            self.headers['Content-Range']=f'bytes {start}-{start+len(data)-1}/{start+len(data)}'

    def read1(self, size=-1):
        self.calls+=1
        if self.fail and self.calls>1: raise OSError('simulated timeout')
        return super().read(size)


class DownloadTests(unittest.TestCase):
    def test_timeout_resumes_and_verified_cache_is_reused(self):
        data=b'a'*(1024*1024)+b'b'*123
        digest=hashlib.sha256(data).hexdigest()
        requests=[]
        def server(request, **kwargs):
            requests.append(request)
            if len(requests)==1:return Response(data,fail=True)
            self.assertEqual(request.get_header('Range'),'bytes=65536-')
            return Response(data[65536:],206,65536)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp, patch('gpu_download.urlopen',side_effect=server), patch('gpu_download.time.sleep'):
            target=Path(temp)/'test.whl'
            self.assertEqual(download('https://example.test',target,digest).read_bytes(),data)
            download('https://example.test',target,digest)
            self.assertEqual(len(requests),2)

    def test_ignored_range_preserves_partial_file(self):
        data=b'correct complete file'
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp, patch('gpu_download.urlopen',return_value=Response(data)):
            target=Path(temp)/'test.whl'; target.with_suffix('.whl.part').write_bytes(b'partial')
            with self.assertRaisesRegex(OSError,'refused Range'):
                download('https://example.test',target,hashlib.sha256(data).hexdigest(),attempts=1)
            self.assertEqual(target.with_suffix('.whl.part').read_bytes(),b'partial')

    def test_bad_checksum_is_never_installed(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp, patch('gpu_download.urlopen',return_value=Response(b'corrupt')):
            target=Path(temp)/'test.whl'
            with self.assertRaisesRegex(ValueError,'Checksum failed'):
                download('https://example.test',target,'0'*64)
            self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
