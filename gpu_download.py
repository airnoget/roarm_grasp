"""Persistent Range downloads with official-index SHA256 verification."""
import hashlib
from http.client import HTTPException
from html.parser import HTMLParser
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k,v in attrs if k == 'href' and v)


def official_wheel(package, version, python_tag):
    index = f'https://download.pytorch.org/whl/cu126/{package}/'
    expected = f'{package}-{version}+cu126-{python_tag}-{python_tag}-win_amd64.whl'
    for attempt in range(5):
        try:
            with urlopen(index, timeout=20) as response:
                html = response.read().decode('utf-8')
            break
        except (OSError, HTTPException):
            if attempt == 4: raise
            print('Retrying official wheel index...', flush=True)
            time.sleep(2)
    parser = Links(); parser.feed(html)
    for href in parser.links:
        parts = urlsplit(urljoin(index, href))
        if unquote(parts.path.rsplit('/',1)[-1]) == expected:
            digest = parse_qs(parts.fragment).get('sha256', [''])[0]
            if not re.fullmatch('[0-9a-fA-F]{64}', digest):
                raise ValueError('Official wheel link has no SHA256')
            return urlunsplit(parts._replace(fragment='')), expected, digest.lower()
    raise ValueError(f'No official wheel for {expected}')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def download(url, destination, digest, attempts=12, timeout=20):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix+'.part')
    if destination.exists():
        if sha256(destination) == digest:
            print('Using verified download:', destination.name, flush=True)
            return destination
        raise ValueError(f'Existing wheel checksum failed: {destination}')
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        if offset and sha256(part) == digest:
            part.replace(destination)
            return destination
        request = Request(url, headers={'Accept-Encoding':'identity', 'User-Agent':'RoArm-GPU-Setup',
                                        **({'Range':f'bytes={offset}-'} if offset else {})})
        try:
            print(f'Downloading {destination.name}: resume at {offset/1e6:.1f} MB', flush=True)
            with urlopen(request, timeout=timeout) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                    if not match or int(match[1]) != offset:
                        raise OSError('Server returned an inconsistent byte range')
                    total = int(match[3])
                elif status == 200:
                    # Never discard gigabytes when a proxy/server ignores Range.
                    if offset:
                        raise OSError('Server refused Range resume; existing partial download preserved')
                    total = int(response.headers.get('Content-Length','0'))
                else:
                    raise OSError(f'Unexpected HTTP status {status}')
                received = offset; last_print = time.monotonic()
                with part.open('ab' if offset else 'wb') as stream:
                    while True:
                        chunk = response.read1(64*1024)
                        if not chunk: break
                        stream.write(chunk); stream.flush(); received += len(chunk)
                        if time.monotonic()-last_print >= 5:
                            print(f'  {received/1e6:.1f}/{total/1e6:.1f} MB', flush=True)
                            last_print = time.monotonic()
                if total and received != total:
                    raise OSError(f'Incomplete response: {received}/{total} bytes')
            if sha256(part) != digest:
                invalid = part.with_name(part.name+f'.invalid-{time.time_ns()}')
                part.replace(invalid)
                raise ValueError(f'Checksum failed; retained invalid file: {invalid}')
            part.replace(destination)
            print('SHA256 verified:', destination.name, flush=True)
            return destination
        except (OSError, HTTPException) as exc:
            print(f'Download interrupted ({attempt+1}/{attempts}): {exc}; partial file retained.', flush=True)
            if attempt+1 == attempts: raise
            time.sleep(2)
    raise RuntimeError('Download retry limit reached')
