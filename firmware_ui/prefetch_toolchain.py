"""Fetch official ESP32-only build archives with range requests and upstream SHA256."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,math
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parent.parent/'.arduino-wifi'
package=json.loads((root/'data/package_esp32_index.upstream.json').read_text(encoding='utf-8'))['packages'][0]
platform=next(x for x in package['platforms'] if x['version']=='3.0.7')
archives=[platform]
for dep in platform['toolsDependencies']:
    if dep['name'] not in ('esp32-arduino-libs','esp-x32'):continue
    tool=next(x for x in package['tools'] if x['name']==dep['name'] and x['version']==dep['version'])
    archives.append(next(x for x in tool['systems'] if x['host']=='x86_64-mingw32'))
out=root/'prefetch';out.mkdir(parents=True,exist_ok=True)

def fetch_archive(a):
    name=a['archiveFileName'];size=int(a['size']);expected=a['checksum'].split(':',1)[1].lower()
    final=out/name
    if final.exists():
        if hashlib.sha256(final.read_bytes()).hexdigest()!=expected:raise ValueError('Existing archive failed hash: '+name)
        return name
    url=a['url'].replace('https://github.com/espressif/','https://dl.espressif.cn/github_assets/espressif/')
    chunk=math.ceil(size/8)
    def part(i):
        start=i*chunk;end=min(size,(i+1)*chunk)-1
        path=out/f'{name}.part{i}'
        if path.exists() and path.stat().st_size==end-start+1:return path
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=120) as response:
                    if response.status!=206 or response.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{end}':
                        raise ValueError('Server returned wrong byte range')
                    with path.open('wb') as stream:
                        while True:
                            data=response.read(256*1024)
                            if not data:break
                            stream.write(data)
                if path.stat().st_size!=end-start+1:raise ValueError('Incomplete byte range')
                print(f'CHUNK_OK {name} {i+1}/8',flush=True)
                return path
            except Exception:
                if attempt==2:raise
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts=list(pool.map(part,range(8)))
    digest=hashlib.sha256()
    with final.open('xb') as stream:
        for path in parts:
            with path.open('rb') as source:
                while True:
                    data=source.read(1024*1024)
                    if not data:break
                    digest.update(data);stream.write(data)
    if digest.hexdigest()!=expected:raise ValueError('Official archive SHA256 mismatch: '+name)
    print(f'ARCHIVE_SHA256_OK {name}',flush=True)
    return name

for a in archives:fetch_archive(a)
