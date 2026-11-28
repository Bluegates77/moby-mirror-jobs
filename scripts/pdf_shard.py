"""Copies open-access papers (direct PDF address) from the deduplicated list of papers we do not hold into a public Hugging Face dataset repo.
Public data only. The token comes from the HF_TOKEN secret. The list is `datasets/openalex-oa-gap/articles/to_fetch_articles_deduplicated.jsonl.gz` in the same repo.
Selection: CLASSES (comm = commercial-use Creative Commons, nc = non-commercial), HOST (case-insensitive part of the host name, empty = all), EXCLUDE (comma list of host parts to leave out; used only when HOST is empty) and a PDF address. Elsevier is always skipped.
A paper is kept only if the bytes start with %PDF and are at most MAXMB. Chunks of CH papers; chunk n belongs to shard n % SHARDS. A chunk whose metadata exists on Hugging Face is skipped, so a re-run resumes.
Speed: WORKERS chunks run at the same time inside one shared pool of CONC connections (PERHOST per site), so one slow site never holds up the rest.
Visibility: a line every 3 minutes (counts, bytes, in flight, staged, last upload) and the same numbers in `_progress/` with every commit. A watchdog ends the job with an error if no paper finishes for 20 minutes, and the job fails if under 20% of what it tried was saved.
Commit-rate cap: the Hub allows 128 commits per hour per repo, so each shard commits at most every SHARDS*3600/TARGET_PER_HOUR seconds."""
import os, json, gzip, io, tarfile, hashlib, time, shutil, datetime, asyncio, re, sys
import aiohttp
from huggingface_hub import HfApi, hf_hub_download
SHARD = int(os.environ['SHARD']); SHARDS = int(os.environ['SHARDS']); CONC = int(os.environ.get('CONC', '300')); PERHOST = int(os.environ.get('PERHOST', '10')); WORKERS = int(os.environ.get('WORKERS', '6'))
CLASSES = os.environ.get('CLASSES', 'comm'); HOST = os.environ.get('HOST', '').lower(); CH = int(os.environ.get('CH', '400')); MAXMB = int(os.environ.get('MAXMB', '80'))
EXCLUDE = [x.strip().lower() for x in os.environ.get('EXCLUDE', '').split(',') if x.strip()]
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
    h = ((r.get('host') or '') + ' ' + r['pdf']).lower(); hn = (r.get('host') or '').lower()
    if 'elsevier' in h or 'sciencedirect' in h: continue
    if HOST and HOST not in hn: continue
    if not HOST and any(x in hn for x in EXCLUDE): continue
    rows.append(r)
rows.sort(key=lambda r: r['id']); nch = -(-len(rows) // CH)
print('selected', len(rows), 'chunks', nch, 'shard', SHARD, '/', SHARDS, 'workers', WORKERS, 'conn', CONC, flush=True)
SAMPLES = {}
S = dict(chunks=0, seen=0, ok=0, bytes=0, skipped={}, inflight=0, last_done=time.time(), last_upload=None, uploads=0, start=time.time())
hostsem = {}
hostpause = {}
def retry_after(resp, i):
    """Seconds the site asked us to wait (Retry-After as seconds or a date); without it, 5, 10, 20, 40 s. Never more than 300 s."""
    v = resp.headers.get('Retry-After')
    if v:
        try: return min(300.0, max(1.0, float(v)))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                return min(300.0, max(1.0, (parsedate_to_datetime(v) - datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
            except Exception: pass
    return 5.0 * 2 ** i
async def fetch(session, sem, r):
    u = r['pdf']; host = re.sub(r'^https?://([^/]+).*', r'\1', u); hs = hostsem.setdefault(host, asyncio.Semaphore(PERHOST))
    why = 'failed'
    for i in range(5):
        wait = hostpause.get(host, 0) - time.time()
        if wait > 0: await asyncio.sleep(wait)
        try:
            async with sem, hs:
                S['inflight'] += 1
                try:
                    async with session.get(u, allow_redirects=True) as resp:
                        if resp.status in (403, 404, 410, 451): return None, 'http %d' % resp.status
                        if resp.status in (202, 429, 502, 503, 504):
                            d = retry_after(resp, i); hostpause[host] = max(hostpause.get(host, 0), time.time() + d); why = 'http %d' % resp.status; S['waits'] = S.get('waits', 0) + 1; continue
                        if resp.status != 200: why = 'http %d' % resp.status; await asyncio.sleep(2 + 2 * i); continue
                        if resp.content_length and resp.content_length > MAXMB << 20: return None, 'too big'
                        b = await resp.read()
                        if len(b) > MAXMB << 20: return None, 'too big'
                        if not b.startswith(b'%PDF'):
                            k = (str(resp.url)[:70] + ' | ' + resp.headers.get('content-type', '?')[:30] + ' | ' + re.sub(r'\s+', ' ', b[:160].decode('utf-8', 'replace'))); SAMPLES[k] = SAMPLES.get(k, 0) + 1
                            return None, 'not a pdf'
                        return b, None
                finally: S['inflight'] -= 1
        except Exception: why = 'failed'; await asyncio.sleep(1 + 2 * i)
    return None, why
async def process(session, sem, ci):
    rs = rows[ci * CH:(ci + 1) * CH]; d = ROOT + '/work/%05d' % ci; os.makedirs(d + '/pdf', exist_ok=True); os.makedirs(d + '/metadata', exist_ok=True)
    out = await asyncio.gather(*[fetch(session, sem, r) for r in rs]); meta = []; skipped = {}; nb = 0
    with tarfile.open(d + '/pdf/chunk_%05d.tar' % ci, 'w') as tf:
        for r, (b, why) in zip(rs, out):
            if b is None: skipped[why] = skipped.get(why, 0) + 1; continue
            name = r['id'].rsplit('/', 1)[-1] + '.pdf'; ti = tarfile.TarInfo(name); ti.size = len(b); ti.mtime = 0; tf.addfile(ti, io.BytesIO(b)); nb += len(b)
            meta.append(dict(openalex=r['id'], doi=r.get('doi'), title=r.get('title'), year=r.get('year'), licence=r['licence'], licence_source='OpenAlex best_oa_location.license', pdf_url=r['pdf'], landing=r.get('landing'), host=r.get('host'), retrieved=T, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b), archive='pdf/chunk_%05d.tar' % ci, member=name))
    bad = len(rs) >= 50 and len(meta) < 0.2 * len(rs)
    if bad:
        os.remove(d + '/pdf/chunk_%05d.tar' % ci); os.makedirs(d + '/failed_chunks', exist_ok=True)
    else:
        with gzip.open(d + '/metadata/chunk_%05d.jsonl.gz' % ci, 'wt') as f:
            for m in meta: f.write(json.dumps(m) + '\n')
    info = dict(bad=bad, chunk=ci, seen=len(rs), copied=len(meta), bytes=nb, skipped=skipped, date=T)
    with open(d + ('/failed_chunks' if bad else '/metadata') + '/chunk_%05d.summary.json' % ci, 'w') as f: json.dump(info, f)
    S['chunks'] += 1; S['seen'] += len(rs); S['ok'] += len(meta); S['bytes'] += nb; S['last_done'] = time.time()
    for k, v in skipped.items(): S['skipped'][k] = S['skipped'].get(k, 0) + v
    return ci, d, info
def line(tag='alive'):
    el = time.time() - S['start']
    return '[%s +%dm] chunks %d papers %d saved %d (%.0f%%) %.1f GB in-flight %d skipped %s uploads %d last upload %s' % (tag, el // 60, S['chunks'], S['seen'], S['ok'], 100.0 * S['ok'] / max(1, S['seen']), S['bytes'] / 1e9, S['inflight'], S['skipped'], S['uploads'], ('%dm ago' % ((time.time() - S['last_upload']) // 60)) if S['last_upload'] else 'none')
async def heartbeat(stop):
    while not stop.is_set():
        try: await asyncio.wait_for(stop.wait(), 180)
        except asyncio.TimeoutError: pass
        print(line(), flush=True)
        for k, v in sorted(SAMPLES.items(), key=lambda x: -x[1])[:3]: print('   non-PDF answer x%d: %s' % (v, k), flush=True)
        if time.time() - S['last_done'] > 1200 and not stop.is_set(): print('WATCHDOG: no chunk finished for 20 minutes', flush=True); os._exit(3)
def flush(stage, cis):
    pd = stage + '/_progress'; os.makedirs(pd, exist_ok=True)
    with open(pd + '/%s_shard%02d.json' % (DEST.replace('/', '_'), SHARD), 'w') as f: json.dump(dict(shard=SHARD, dest=DEST, time=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), **{k: v for k, v in S.items() if k not in ('last_done', 'start')}), f)
    if DRY: print('DRY: would upload', cis, flush=True); return
    for t in range(8):
        try:
            a.upload_folder(repo_id=R, repo_type='dataset', folder_path=stage, path_in_repo=DEST, commit_message='Open-access papers %s chunks %s (shard %d)' % (DEST.rsplit('/', 1)[-1], cis, SHARD)); print('uploaded', cis, flush=True); return
        except Exception as e: print('upload retry', str(e)[:120], flush=True); time.sleep(60)
    print('UPLOAD FAILED', flush=True); os._exit(4)
def size_of(p): return sum(os.path.getsize(os.path.join(q, f)) for q, _, fs in os.walk(p) for f in fs)
async def main():
    loop = asyncio.get_running_loop(); shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT + '/work')
    sem = asyncio.Semaphore(CONC); stop = asyncio.Event(); st = dict(n=0, pending=[], dir=ROOT + '/stage0'); os.makedirs(st['dir'])
    mine = [ci for ci in range(nch) if ci % SHARDS == SHARD]
    if LIMIT: mine = mine[:LIMIT]
    todo = []
    for ci in mine:
        if not DRY and await loop.run_in_executor(None, lambda: a.file_exists(R, DEST + '/metadata/chunk_%05d.jsonl.gz' % ci, repo_type='dataset')): print('skip', ci, flush=True); continue
        todo.append(ci)
    print('to do', len(todo), 'of', len(mine), flush=True)
    last_flush = time.time() - MIN_GAP + MIN_GAP * SHARD / SHARDS
    conn = aiohttp.TCPConnector(limit=CONC, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=90, sock_connect=20, sock_read=40), headers={'User-Agent': 'moby-open-access-copy (public Creative Commons papers)'}, trust_env=True) as session:
        hb = asyncio.create_task(heartbeat(stop)); work = list(todo)
        async def worker():
            while work:
                if size_of(ROOT) > 9e9: await asyncio.sleep(10); continue
                ci = work.pop(0); ci, d, info = await process(session, sem, ci)
                for sub in ('pdf', 'metadata', 'failed_chunks'):
                    if not os.path.isdir(d + '/' + sub): continue
                    for f in os.listdir(d + '/' + sub): os.makedirs(st['dir'] + '/' + sub, exist_ok=True); shutil.move(d + '/' + sub + '/' + f, st['dir'] + '/' + sub + '/' + f)
                shutil.rmtree(d, ignore_errors=True); st['pending'].append(ci)
                print('chunk', ci, info['copied'], 'of', info['seen'], '%.0f MB' % (info['bytes'] / 1e6), info['skipped'], flush=True)
        ws = [asyncio.create_task(worker()) for _ in range(WORKERS)]
        while True:
            finished = all(w.done() for w in ws)
            if st['pending'] and (time.time() - last_flush >= MIN_GAP or finished):
                wait = MIN_GAP - (time.time() - last_flush)
                if finished and wait > 0 and not DRY: print('waiting %.0fs (rate cap)' % wait, flush=True); await asyncio.sleep(wait)
                stage, cis = st['dir'], list(st['pending']); st['n'] += 1; st['dir'] = ROOT + '/stage%d' % st['n']; os.makedirs(st['dir']); st['pending'] = []
                await loop.run_in_executor(None, flush, stage, cis); last_flush = time.time(); S['last_upload'] = last_flush; S['uploads'] += 1
                if not DRY: shutil.rmtree(stage, ignore_errors=True)
            elif finished: break
            else: await asyncio.sleep(5)
        for w in ws:
            if w.exception(): raise w.exception()
        stop.set(); await hb
    print(line('DONE'), flush=True)
    if S['seen'] > 200 and S['ok'] < 0.2 * S['seen']: print('FAILED: under 20% saved', flush=True); sys.exit(1)
    print('SHARD DONE', SHARD, flush=True)
asyncio.run(main())
