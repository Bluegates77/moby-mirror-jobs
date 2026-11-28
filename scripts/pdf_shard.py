"""Copies open-access papers (direct PDF address) from the deduplicated list of papers we do not hold into a public Hugging Face dataset repo.
Public data only. The token comes from the HF_TOKEN secret. The list is `datasets/openalex-oa-gap/articles/to_fetch_articles_deduplicated.jsonl.gz` in the same repo.
Selection: CLASSES (comm = commercial-use Creative Commons, nc = non-commercial), HOST (case-insensitive part of the publisher/host name, empty = all), and a PDF address.
Elsevier is always skipped (its terms do not allow it; see RESEARCH.md). A paper is kept only if the bytes start with %PDF and are at most MAXMB.
Chunks of CH papers (in list order after filtering); chunk n belongs to shard n % SHARDS. A chunk whose metadata file exists on Hugging Face is skipped, so a re-run is safe.
Each paper gets a metadata row with the licence evidence (OpenAlex id, DOI, licence code, address it came from, host, date retrieved, sha256, bytes).
Commit-rate cap as in pmc_shard.py (128 commits per hour per repo)."""
import os, json, gzip, io, tarfile, hashlib, time, shutil, datetime, asyncio, re
import aiohttp
from huggingface_hub import HfApi, hf_hub_download
SHARD = int(os.environ['SHARD']); SHARDS = int(os.environ['SHARDS']); CONC = int(os.environ.get('CONC', '64')); PERHOST = int(os.environ.get('PERHOST', '6'))
CLASSES = os.environ.get('CLASSES', 'comm'); HOST = os.environ.get('HOST', '').lower(); CH = int(os.environ.get('CH', '400')); MAXMB = int(os.environ.get('MAXMB', '80'))
LIMIT = int(os.environ.get('LIMIT_CHUNKS', '0')); DRY = os.environ.get('DRY') == '1'
TARGET = float(os.environ.get('TARGET_PER_HOUR', '100')); MIN_GAP = SHARDS * 3600.0 / TARGET
R = 'Middle-child/moby-public-assets'; DEST = os.environ['DEST']
a = HfApi(); T = str(datetime.date.today()); ROOT = os.environ.get('RUNNER_TEMP', '/tmp') + '/pdf%d' % SHARD
LIST = 'datasets/openalex-oa-gap/articles/to_fetch_articles_deduplicated.jsonl.gz'
lp = hf_hub_download(R, LIST, repo_type='dataset')
rows = []
for l in gzip.open(lp, 'rt'):
    r = json.loads(l)
    if not r.get('pdf') or r.get('class_') != CLASSES: continue
    h = ((r.get('host') or '') + ' ' + r['pdf']).lower()
    if 'elsevier' in h or 'sciencedirect' in h: continue
    if HOST and HOST not in (r.get('host') or '').lower(): continue
    rows.append(r)
rows.sort(key=lambda r: r['id']); nch = -(-len(rows) // CH)
print('selected', len(rows), 'chunks', nch, 'shard', SHARD, '/', SHARDS, flush=True)
hostsem = {}
async def fetch(session, sem, r):
    u = r['pdf']; hs = hostsem.setdefault(re.sub(r'^https?://([^/]+).*', r'\1', u), asyncio.Semaphore(PERHOST))
    for i in range(3):
        try:
            async with sem, hs:
                async with session.get(u, allow_redirects=True) as resp:
                    if resp.status in (403, 404, 410, 451): return None, 'http %d' % resp.status
                    if resp.status != 200: await asyncio.sleep(2 + 3 * i); continue
                    if resp.content_length and resp.content_length > MAXMB << 20: return None, 'too big'
                    b = await resp.read()
                    if len(b) > MAXMB << 20: return None, 'too big'
                    if resp.content_length and len(b) != resp.content_length and not resp.headers.get('Content-Encoding'): await asyncio.sleep(2 + 3 * i); continue
                    if not b.startswith(b'%PDF'): return None, 'not a pdf'
                    return b, None
        except Exception as e: await asyncio.sleep(2 + 3 * i); err = type(e).__name__
    return None, 'failed'
async def process(session, sem, ci, d):
    rs = rows[ci * CH:(ci + 1) * CH]; os.makedirs(d + '/pdf', exist_ok=True); os.makedirs(d + '/metadata', exist_ok=True)
    out = await asyncio.gather(*[fetch(session, sem, r) for r in rs]); meta = []; skipped = {}; nb = 0
    tarp = d + '/pdf/chunk_%05d.tar' % ci
    with tarfile.open(tarp, 'w') as tf:
        for r, (b, why) in zip(rs, out):
            if b is None: skipped[why] = skipped.get(why, 0) + 1; continue
            name = r['id'].rsplit('/', 1)[-1] + '.pdf'; ti = tarfile.TarInfo(name); ti.size = len(b); ti.mtime = 0; tf.addfile(ti, io.BytesIO(b)); nb += len(b)
            meta.append(dict(openalex=r['id'], doi=r.get('doi'), title=r.get('title'), year=r.get('year'), licence=r['licence'], licence_source='OpenAlex best_oa_location.license', pdf_url=r['pdf'], landing=r.get('landing'), host=r.get('host'), retrieved=T, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b), archive='pdf/chunk_%05d.tar' % ci, member=name))
    with gzip.open(d + '/metadata/chunk_%05d.jsonl.gz' % ci, 'wt') as f:
        for m in meta: f.write(json.dumps(m) + '\n')
    info = dict(chunk=ci, seen=len(rs), copied=len(meta), bytes=nb, skipped=skipped, date=T)
    with open(d + '/metadata/chunk_%05d.summary.json' % ci, 'w') as f: json.dump(info, f)
    return info
def flush(d, cis):
    if DRY: print('DRY: would upload', cis, flush=True); return
    for t in range(8):
        try: a.upload_folder(repo_id=R, repo_type='dataset', folder_path=d, path_in_repo=DEST, commit_message='Open-access papers %s chunks %s (shard %d)' % (DEST.rsplit('/', 1)[-1], cis, SHARD)); print('uploaded', cis, flush=True); return
        except Exception as e: print('upload retry', str(e)[:120], flush=True); time.sleep(60)
    raise SystemExit('upload failed')
async def main():
    loop = asyncio.get_running_loop(); shutil.rmtree(ROOT, ignore_errors=True)
    sem = asyncio.Semaphore(CONC); pending = []; n = 0; d = ROOT + '/stage0'; os.makedirs(d); up = None; updir = None
    last_flush = time.time() - MIN_GAP + MIN_GAP * SHARD / SHARDS
    conn = aiohttp.TCPConnector(limit=CONC, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=180), headers={'User-Agent': 'moby-open-access-copy (public Creative Commons papers)'}, trust_env=True) as session:
        mine = [ci for ci in range(nch) if ci % SHARDS == SHARD]
        if LIMIT: mine = mine[:LIMIT]
        for ci in mine:
            if not DRY and await loop.run_in_executor(None, lambda: a.file_exists(R, DEST + '/metadata/chunk_%05d.jsonl.gz' % ci, repo_type='dataset')): print('skip', ci, flush=True); continue
            t0 = time.time(); info = await process(session, sem, ci, d); pending.append(ci)
            print('chunk', ci, info['copied'], 'of', info['seen'], '%.0f MB' % (info['bytes'] / 1e6), info['skipped'], 'in %.0fs' % (time.time() - t0), flush=True)
            if (time.time() - last_flush >= MIN_GAP or sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(d) for f in fs) > 3e9) and pending:
                if up is not None: await up; (None if DRY else shutil.rmtree(updir, ignore_errors=True))
                up = loop.run_in_executor(None, flush, d, list(pending)); updir = d; last_flush = time.time(); n += 1; d = ROOT + '/stage%d' % n; os.makedirs(d); pending = []
        if up is not None: await up; (None if DRY else shutil.rmtree(updir, ignore_errors=True))
        if pending:
            wait = MIN_GAP - (time.time() - last_flush)
            if wait > 0 and not DRY: print('waiting %.0fs (rate cap)' % wait, flush=True); await asyncio.sleep(wait)
            await loop.run_in_executor(None, flush, d, list(pending))
    print('SHARD DONE', SHARD, flush=True)
asyncio.run(main())
