"""Copies one shard of the PubMed Central Open Access commercial-use subset into a public Hugging Face dataset repo.
Public data only. The token comes from the HF_TOKEN secret (never in this repository).
Chunks are 40,000 article versions of the PMC inventory (same chunk numbers as the first local run); chunk n belongs to shard n % SHARDS.
A chunk whose metadata file already exists on Hugging Face is skipped, so the job can be re-run safely.
Terms: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ (CC0, CC BY, CC BY-SA, CC BY-ND = commercial use allowed) and the README of s3://pmc-oa-opendata."""
import os, re, json, gzip, io, tarfile, hashlib, time, shutil, threading, collections, datetime, concurrent.futures as cf
import requests
from huggingface_hub import HfApi, hf_hub_download
SHARD = int(os.environ['SHARD']); SHARDS = int(os.environ['SHARDS']); FLUSH = int(os.environ.get('FLUSH_CHUNKS', '3')); THREADS = int(os.environ.get('THREADS', '48'))
R = 'Middle-child/moby-public-assets'; DEST = 'datasets/pmc-oa-commercial-use'; B = 'https://pmc-oa-opendata.s3.amazonaws.com/'; CH = 40000
OK = {'CC0', 'CC BY', 'CC BY-SA', 'CC BY-ND'}
a = HfApi(); T = str(datetime.date.today()); ROOT = os.environ.get('RUNNER_TEMP', '/tmp') + '/pmc%d' % SHARD
_tl = threading.local()
def get(u, tries=6):
    for i in range(tries):
        try:
            s = getattr(_tl, 's', None)
            if s is None:
                s = _tl.s = requests.Session(); s.mount('https://', requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)); s.headers['User-Agent'] = 'moby-hf-copy'
            r = s.get(u, timeout=60)
            if r.status_code == 404: return None
            if r.status_code == 200: return r.content
            time.sleep(1 + i)
        except Exception:
            _tl.s = None; time.sleep(1 + i)
    raise Exception('get failed ' + u)
kp = hf_hub_download(R, DEST + '/inventory/pmc_keys.txt.gz', repo_type='dataset')
keys = gzip.open(kp, 'rt').read().split('\n'); nchunks = -(-len(keys) // CH)
print('keys', len(keys), 'chunks', nchunks, 'shard', SHARD, '/', SHARDS, flush=True)
def exists(ci):
    return a.file_exists(R, DEST + '/metadata/chunk_%04d.jsonl.gz' % ci, repo_type='dataset')
def process(ci, d):
    ks = keys[ci * CH:(ci + 1) * CH]
    def meta(k):
        b = get(B + 'metadata/' + k + '.json'); return json.loads(b) if b else None
    with cf.ThreadPoolExecutor(THREADS) as ex: ms = list(ex.map(meta, ks))
    lic = collections.Counter(); sel = []
    for k, m in zip(ks, ms):
        if m is None: lic['(no metadata)'] += 1; continue
        lic[(m.get('license_code') or '(none)') + ('' if m.get('is_pmc_openaccess') else ' [not OA subset]')] += 1
        if m.get('is_pmc_openaccess') and m.get('license_code') in OK and m.get('xml_url'): sel.append((k, m))
    os.makedirs(d + '/xml', exist_ok=True); os.makedirs(d + '/metadata', exist_ok=True)
    def xml(km):
        k, m = km; want = re.search(r'md5=([0-9a-f]{32})', m['xml_url']).group(1)
        for t in range(4):
            b = get(B + k + '/' + k + '.xml')
            if b is not None and hashlib.md5(b).hexdigest() == want: return k, m, b
        raise Exception('xml md5 mismatch ' + k)
    def gen():
        with cf.ThreadPoolExecutor(THREADS) as ex:
            for i in range(0, len(sel), 1500):
                for x in ex.map(xml, sel[i:i + 1500]): yield x
    tarp = d + '/xml/chunk_%04d.tar.gz' % ci; rows = []; nbytes = 0
    with tarfile.open(tarp, 'w:gz', compresslevel=3) as tf:
        for k, m, b in gen():
            ti = tarfile.TarInfo(k + '.xml'); ti.size = len(b); ti.mtime = 0; tf.addfile(ti, io.BytesIO(b)); nbytes += len(b)
            rows.append(dict(pmcid=m['pmcid'], version=m['version'], pmid=m.get('pmid'), doi=m.get('doi'), title=m.get('title'), citation=m.get('citation'), license_code=m['license_code'], is_retracted=m.get('is_retracted'), xml_md5=re.search(r'md5=([0-9a-f]{32})', m['xml_url']).group(1), archive='xml/chunk_%04d.tar.gz' % ci, member=k + '.xml'))
    with gzip.open(d + '/metadata/chunk_%04d.jsonl.gz' % ci, 'wt') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    sh = hashlib.sha256()
    with open(tarp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): sh.update(c)
    info = dict(chunk=ci, first=ks[0], last=ks[-1], seen=len(ks), copied=len(rows), xml_bytes=nbytes, archive_size=os.path.getsize(tarp), archive_sha256=sh.hexdigest(), licences=dict(lic), date=T)
    with open(d + '/metadata/chunk_%04d.summary.json' % ci, 'w') as f: json.dump(info, f)
    return info
def flush(d, cis):
    for t in range(8):
        try:
            a.upload_folder(repo_id=R, repo_type='dataset', folder_path=d, path_in_repo=DEST, commit_message='PMC OA commercial-use chunks %s (shard %d)' % (cis, SHARD)); return
        except Exception as e:
            print('upload retry', str(e)[:120], flush=True); time.sleep(60)
    raise SystemExit('upload failed')
pending = []; d = ROOT + '/stage0'; n = 0
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(d)
for ci in range(nchunks):
    if ci % SHARDS != SHARD: continue
    if exists(ci): print('skip', ci, flush=True); continue
    t0 = time.time(); info = process(ci, d); pending.append(ci)
    print('chunk', ci, info['copied'], 'of', info['seen'], 'in %.0fs' % (time.time() - t0), flush=True)
    if len(pending) >= FLUSH:
        flush(d, pending); shutil.rmtree(d, ignore_errors=True); n += 1; d = ROOT + '/stage%d' % n; os.makedirs(d); pending = []
if pending: flush(d, pending)
print('SHARD DONE', SHARD, flush=True)
