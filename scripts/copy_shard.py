"""Async (aiohttp) shard worker for public-data copies into the public Hugging Face dataset repo. JOB=openneuro or JOB=vhp.
Token only from the HF_TOKEN secret. SHARD/SHARDS split the work; a commit cap spreads each shard's commits (TARGET_PER_HOUR in total).
openneuro: CC0 MRI datasets (selection list on Hugging Face), one uncompressed tar per dataset, files checked by size and (single-part ETag) md5.
vhp: NLM Visible Human Project, PNG_format directories of Male-Images and Female-Images (file list on Hugging Face), files checked by size."""
import os, re, json, html, hashlib, time, tarfile, shutil, datetime, asyncio, urllib.parse
import aiohttp
from huggingface_hub import HfApi, hf_hub_download
JOB = os.environ['JOB']; SHARD = int(os.environ['SHARD']); SHARDS = int(os.environ['SHARDS']); CONC = int(os.environ.get('CONC', '128'))
TARGET = float(os.environ.get('TARGET_PER_HOUR', '60')); MIN_GAP = SHARDS * 3600.0 / TARGET; ONLY = os.environ.get('ONLY'); DRY = os.environ.get('DRY') == '1'
R = 'Middle-child/moby-public-assets'; a = HfApi(); T = str(datetime.date.today()); W = os.environ.get('RUNNER_TEMP', '/tmp') + '/cs%d' % SHARD
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): h.update(c)
    return h.hexdigest()
async def fetch(session, sem, url, path, size=None, tries=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(tries):
        try:
            async with sem:
                async with session.get(url) as r:
                    if r.status != 200: raise Exception('status %d' % r.status)
                    with open(path, 'wb') as f:
                        async for c in r.content.iter_chunked(1 << 20): f.write(c)
            if size is None or os.path.getsize(path) == size: return True
        except Exception: pass
        await asyncio.sleep(min(1 + i, 6))
    return False
async def text(session, sem, url, tries=6):
    for i in range(tries):
        try:
            async with sem:
                async with session.get(url) as r:
                    if r.status == 200: return await r.text()
        except Exception: pass
        await asyncio.sleep(min(1 + i, 6))
    raise Exception('get failed ' + url)
def flush(d, label, dest):
    if DRY: print('DRY upload', label, flush=True); return
    for t in range(8):
        try: a.upload_folder(repo_id=R, repo_type='dataset', folder_path=d, path_in_repo=dest, commit_message='%s (shard %d)' % (label, SHARD)); print('uploaded', label, flush=True); return
        except Exception as e: print('upload retry', str(e)[:100], flush=True); time.sleep(60)
    raise SystemExit('upload failed')
async def openneuro(session, sem):
    sel = json.load(open(hf_hub_download(R, 'datasets/openneuro/selection.json', repo_type='dataset')))
    mine = [s for i, s in enumerate(sel) if i % SHARDS == SHARD and (not ONLY or s['id'] == ONLY)]
    print('openneuro datasets for this shard:', len(mine), flush=True)
    S3 = 'https://s3.amazonaws.com/openneuro.org/'; loop = asyncio.get_running_loop()
    stage = W + '/stage'; pend = []; last = time.time() - MIN_GAP + MIN_GAP * SHARD / SHARDS; up = None; os.makedirs(stage, exist_ok=True)
    for s in mine:
        ds = s['id']
        if not DRY and a.file_exists(R, 'datasets/openneuro/%s.tar' % ds, repo_type='dataset'): print('skip', ds, flush=True); continue
        keys = []; tok = None
        while True:
            u = S3 + '?list-type=2&prefix=%s/' % ds + ('&continuation-token=' + urllib.parse.quote(tok) if tok else '')
            t = await text(session, sem, u)
            for m in re.finditer(r'<Contents><Key>(.*?)</Key>.*?<ETag>&quot;(.*?)&quot;</ETag><Size>(\d+)</Size>', t, re.S): keys.append((html.unescape(m.group(1)), m.group(2), int(m.group(3))))
            tk = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', t)
            if not tk: break
            tok = html.unescape(tk.group(1))
        tmp = W + '/tmp_' + ds; shutil.rmtree(tmp, ignore_errors=True); skipped = []
        async def one(k):
            key, etag, size = k; p = tmp + '/' + key
            ok = await fetch(session, sem, S3 + urllib.parse.quote(key), p, size)
            md5 = None
            if ok and '-' not in etag:
                m = hashlib.md5()
                with open(p, 'rb') as f:
                    for c in iter(lambda: f.read(1 << 22), b''): m.update(c)
                md5 = m.hexdigest() == etag
                if not md5: ok = False
            if not ok: skipped.append(dict(key=key, size=size, reason='size or md5 mismatch after retries'))
            return md5
        res = await asyncio.gather(*[one(k) for k in keys])
        tp = stage + '/openneuro/%s.tar' % ds; os.makedirs(os.path.dirname(tp), exist_ok=True)
        with tarfile.open(tp, 'w') as tf: tf.add(tmp + '/' + ds, arcname=ds) if os.path.isdir(tmp + '/' + ds) else None
        shutil.rmtree(tmp, ignore_errors=True)
        info = dict(id=ds, name=s['name'], licence=s['licence'], files=len(keys), bytes=sum(k[2] for k in keys), md5_checked=sum(1 for r in res if r), skipped=skipped, archive_size=os.path.getsize(tp), archive_sha256=sha(tp), path='datasets/openneuro/%s.tar' % ds, date=T)
        os.makedirs(stage + '/index', exist_ok=True); json.dump(info, open(stage + '/index/%s.json' % ds, 'w'))
        pend.append(ds); print('dataset', ds, info['files'], 'files', info['bytes'], 'bytes, skipped', len(skipped), flush=True)
        if time.time() - last >= MIN_GAP:
            if up is not None: await up
            d = stage; up = loop.run_in_executor(None, flush, d, 'OpenNeuro CC0 datasets ' + ','.join(pend), 'datasets/openneuro'); last = time.time(); pend = []
            stage = W + '/stage%d' % int(time.time()); os.makedirs(stage, exist_ok=True)
    if up is not None: await up
    if pend:
        wait = MIN_GAP - (time.time() - last)
        if wait > 0 and not DRY: await asyncio.sleep(wait)
        await loop.run_in_executor(None, flush, stage, 'OpenNeuro CC0 datasets ' + ','.join(pend), 'datasets/openneuro')
async def vhp(session, sem):
    files = json.load(open(hf_hub_download(R, 'datasets/visible-human/_filelist.json', repo_type='dataset')))
    VH = 'https://data.lhncbc.nlm.nih.gov/public/Visible-Human/'
    mine = [f for i, f in enumerate(files) if i % SHARDS == SHARD and (not ONLY or f[0].startswith(ONLY))]
    print('visible human files for this shard:', len(mine), flush=True)
    root = W + '/vh'; os.makedirs(root, exist_ok=True); rows = []; bad = []
    async def one(f):
        p, s = f; ok = await fetch(session, sem, VH + urllib.parse.quote(p), root + '/' + p, s)
        if ok: rows.append(dict(path='datasets/visible-human/' + p, size=s, sha256=sha(root + '/' + p), url=VH + p, date=T))
        else: bad.append(p)
    for i in range(0, len(mine), 300): await asyncio.gather(*[one(f) for f in mine[i:i + 300]])
    os.makedirs(root + '/_index', exist_ok=True); json.dump(dict(shard=SHARD, files=rows, failed=bad), open(root + '/_index/shard_%02d.json' % SHARD, 'w'))
    if SHARD == 0: open(root + '/CREDITS.txt', 'w').write('Visible Human Project (NLM), Male-Images and Female-Images, PNG_format directories, downloaded %s from https://data.lhncbc.nlm.nih.gov/public/Visible-Human/ . NLM (https://www.nlm.nih.gov/research/visible/visible_human.html): "a public-domain library of cross-sectional cryosection, CT, and MRI images obtained from one male cadaver and one female cadaver"; the data READMEs say usage is approved under the NLM Terms and Conditions (https://www.nlm.nih.gov/databases/download/terms_and_conditions.html): acknowledge "Courtesy of the U.S. National Library of Medicine"; redistributors keep the data current or say plainly it may not be. Unmodified; each file size checked against NLM\'s index. NOT copied: the 70mm and Fullcolor raw directories (duplicates of the PNG data), the radiological directory (not requested) and "Visible Human 2.0 - Peter Ratiu" (separate authorship, terms not read).\n' % T)
    print('failed files:', len(bad), flush=True)
    await asyncio.get_running_loop().run_in_executor(None, flush, root, 'Visible Human PNG files', 'datasets/visible-human')
async def main():
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    conn = aiohttp.TCPConnector(limit=CONC, limit_per_host=CONC, ttl_dns_cache=300); sem = asyncio.Semaphore(CONC)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=300), headers={'User-Agent': 'moby-hf-copy'}, trust_env=True) as session:
        await (openneuro if JOB == 'openneuro' else vhp)(session, sem)
    print('SHARD DONE', JOB, SHARD, flush=True)
asyncio.run(main())
