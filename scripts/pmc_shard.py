"""Copies one shard of the PubMed Central Open Access commercial-use subset into a public Hugging Face dataset repo (async version).
Public data only. The token comes from the HF_TOKEN secret (never in this repository).
Chunks are 40,000 article versions of the PMC inventory (same chunk numbers as the first local run); chunk n belongs to shard n % SHARDS.
A chunk whose metadata file already exists on Hugging Face is skipped, so the job can be re-run safely.
Commit-rate cap: the Hub allows 128 commits per hour per repo. With SHARDS jobs and TARGET_PER_HOUR commits per hour in total, each shard waits at least
SHARDS*3600/TARGET_PER_HOUR seconds between commits (unless it has nothing more to do), so the total can never pass TARGET_PER_HOUR.
Terms: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ (CC0, CC BY, CC BY-SA, CC BY-ND = commercial use allowed) and the README of s3://pmc-oa-opendata."""
import os, re, json, gzip, io, tarfile, hashlib, time, shutil, collections, datetime, asyncio, queue, threading
import aiohttp
from huggingface_hub import HfApi, hf_hub_download
SHARD = int(os.environ['SHARD']); SHARDS = int(os.environ['SHARDS']); CONC = int(os.environ.get('CONC', '256'))
TARGET = float(os.environ.get('TARGET_PER_HOUR', '112')); MIN_GAP = SHARDS * 3600.0 / TARGET
DRY = os.environ.get('DRY') == '1'
R = 'Middle-child/moby-public-assets'; DEST = 'datasets/pmc-oa-commercial-use'; B = 'https://pmc-oa-opendata.s3.amazonaws.com/'; CH = int(os.environ.get('CH', '40000'))
OK = {'CC0', 'CC BY', 'CC BY-SA', 'CC BY-ND'}
a = HfApi(); T = str(datetime.date.today()); ROOT = os.environ.get('RUNNER_TEMP', '/tmp') + '/pmc%d' % SHARD
kp = hf_hub_download(R, DEST + '/inventory/pmc_keys.txt.gz', repo_type='dataset')
keys = gzip.open(kp, 'rt').read().split('\n'); nchunks = -(-len(keys) // CH)
print('keys', len(keys), 'chunks', nchunks, 'shard', SHARD, '/', SHARDS, 'min gap between commits %.0fs' % MIN_GAP, flush=True)
async def aget(session, sem, url, tries=6):
    for i in range(tries):
        try:
            async with sem:
                async with session.get(url) as r:
                    if r.status == 404: return None
                    if r.status == 200: return await r.read()
        except Exception: pass
        await asyncio.sleep(min(1 + i, 6))
    raise Exception('get failed ' + url)
def write_batch(tf, batch, rows, ci):
    n = 0
    for k, m, b in batch:
        ti = tarfile.TarInfo(k + '.xml'); ti.size = len(b); ti.mtime = 0; tf.addfile(ti, io.BytesIO(b)); n += len(b)
        rows.append(dict(pmcid=m['pmcid'], version=m['version'], pmid=m.get('pmid'), doi=m.get('doi'), title=m.get('title'), citation=m.get('citation'), license_code=m['license_code'], is_retracted=m.get('is_retracted'), xml_md5=re.search(r'md5=([0-9a-f]{32})', m['xml_url']).group(1), archive='xml/chunk_%04d.tar.gz' % ci, member=k + '.xml'))
    return n
async def process(session, sem, ci, d):
    ks = keys[ci * CH:(ci + 1) * CH]
    async def meta(k):
        b = await aget(session, sem, B + 'metadata/' + k + '.json'); return json.loads(b) if b else None
    ms = []
    for i in range(0, len(ks), 5000): ms += await asyncio.gather(*[meta(k) for k in ks[i:i + 5000]])
    lic = collections.Counter(); sel = []
    for k, m in zip(ks, ms):
        if m is None: lic['(no metadata)'] += 1; continue
        lic[(m.get('license_code') or '(none)') + ('' if m.get('is_pmc_openaccess') else ' [not OA subset]')] += 1
        if m.get('is_pmc_openaccess') and m.get('license_code') in OK and m.get('xml_url'): sel.append((k, m))
    os.makedirs(d + '/xml', exist_ok=True); os.makedirs(d + '/metadata', exist_ok=True)
    async def xml(km):
        k, m = km; want = re.search(r'md5=([0-9a-f]{32})', m['xml_url']).group(1)
        for t in range(4):
            b = await aget(session, sem, B + k + '/' + k + '.xml')
            if b is not None and hashlib.md5(b).hexdigest() == want: return k, m, b
        raise Exception('xml md5 mismatch ' + k)
    tarp = d + '/xml/chunk_%04d.tar.gz' % ci; rows = []; nbytes = 0; loop = asyncio.get_running_loop(); wt = None
    with tarfile.open(tarp, 'w:gz', compresslevel=3) as tf:
        for i in range(0, len(sel), 2000):
            batch = await asyncio.gather(*[xml(x) for x in sel[i:i + 2000]])
            if wt is not None: nbytes += await wt
            wt = loop.run_in_executor(None, write_batch, tf, batch, rows, ci)
        if wt is not None: nbytes += await wt
    with gzip.open(d + '/metadata/chunk_%04d.jsonl.gz' % ci, 'wt') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    sh = hashlib.sha256()
    with open(tarp, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): sh.update(c)
    info = dict(chunk=ci, first=ks[0], last=ks[-1], seen=len(ks), copied=len(rows), xml_bytes=nbytes, archive_size=os.path.getsize(tarp), archive_sha256=sh.hexdigest(), licences=dict(lic), date=T)
    with open(d + '/metadata/chunk_%04d.summary.json' % ci, 'w') as f: json.dump(info, f)
    return info
def flush(d, cis):
    if DRY: print('DRY: would upload', cis, flush=True); return
    for t in range(8):
        try:
            a.upload_folder(repo_id=R, repo_type='dataset', folder_path=d, path_in_repo=DEST, commit_message='PMC OA commercial-use chunks %s (shard %d)' % (cis, SHARD)); print('uploaded', cis, flush=True); return
        except Exception as e:
            print('upload retry', str(e)[:120], flush=True); time.sleep(60)
    raise SystemExit('upload failed')
async def main():
    loop = asyncio.get_running_loop()
    shutil.rmtree(ROOT, ignore_errors=True)
    conn = aiohttp.TCPConnector(limit=CONC, limit_per_host=CONC, ttl_dns_cache=300)
    sem = asyncio.Semaphore(CONC)
    pending = []; n = 0; d = ROOT + '/stage0'; os.makedirs(d); last_flush = 0.0; up = None; updir = None
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=120), headers={'User-Agent': 'moby-hf-copy'}, trust_env=True) as session:
        mine = [ci for ci in range(nchunks) if ci % SHARDS == SHARD]
        for idx, ci in enumerate(mine):
            if not DRY and await loop.run_in_executor(None, lambda: a.file_exists(R, DEST + '/metadata/chunk_%04d.jsonl.gz' % ci, repo_type='dataset')): print('skip', ci, flush=True); continue
            t0 = time.time(); info = await process(session, sem, ci, d); pending.append(ci)
            print('chunk', ci, info['copied'], 'of', info['seen'], 'in %.0fs' % (time.time() - t0), flush=True)
            if (time.time() - last_flush >= MIN_GAP) and pending:
                if up is not None: await up; shutil.rmtree(updir, ignore_errors=True)
                up = loop.run_in_executor(None, flush, d, list(pending)); updir = d; last_flush = time.time()
                n += 1; d = ROOT + '/stage%d' % n; os.makedirs(d); pending = []
        if up is not None: await up; shutil.rmtree(updir, ignore_errors=True)
        if pending:
            wait = MIN_GAP - (time.time() - last_flush)
            if wait > 0 and not DRY: print('waiting %.0fs before the last commit (rate cap)' % wait, flush=True); await asyncio.sleep(wait)
            await loop.run_in_executor(None, flush, d, list(pending))
    print('SHARD DONE', SHARD, flush=True)
asyncio.run(main())
