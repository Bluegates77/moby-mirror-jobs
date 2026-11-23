"""Async (aiohttp) shard worker for public-data copies into the public Hugging Face dataset repo. JOB=openneuro or JOB=vhp.
Token only from the HF_TOKEN secret. SHARD/SHARDS split the work; a commit cap spreads each shard's commits (TARGET_PER_HOUR in total).
openneuro: CC0 MRI datasets (selection list on Hugging Face), one uncompressed tar per dataset, files checked by size and (single-part ETag) md5.
vhp: NLM Visible Human Project, PNG_format directories of Male-Images and Female-Images (file list on Hugging Face), files checked by size."""
import os, re, json, html, hashlib, time, tarfile, shutil, datetime, asyncio, urllib.parse, collections
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
def flush_rm(d, label, dest):
    flush(d, label, dest); shutil.rmtree(d, ignore_errors=True)
async def openneuro(session, sem):
    sel = json.load(open(hf_hub_download(R, 'datasets/openneuro/selection.json', repo_type='dataset')))
    REDO = [x for x in os.environ.get('REDO', '').split(',') if x]   # datasets to copy again and overwrite (their first copy lost files to the old 300 s limit)
    mine = [s for i, s in enumerate(sel) if i % SHARDS == SHARD and (not ONLY or s['id'] == ONLY) and (not REDO or s['id'] in REDO)]
    print('openneuro datasets for this shard:', len(mine), flush=True)
    S3 = 'https://s3.amazonaws.com/openneuro.org/'; loop = asyncio.get_running_loop()
    stage = W + '/stage'; staged = 0; pend = []; last = time.time() - MIN_GAP + MIN_GAP * SHARD / SHARDS; up = None; os.makedirs(stage, exist_ok=True)
    for s in mine:
        ds = s['id']
        if not DRY and not REDO and (a.file_exists(R, 'datasets/openneuro/openneuro/%s.tar' % ds, repo_type='dataset') or a.file_exists(R, 'datasets/openneuro/%s.tar' % ds, repo_type='dataset')): print('skip', ds, flush=True); continue
        keys = []; tok = None
        while True:
            u = S3 + '?list-type=2&prefix=%s/' % ds + ('&continuation-token=' + urllib.parse.quote(tok) if tok else '')
            t = await text(session, sem, u)
            for blk in re.findall(r'<Contents>(.*?)</Contents>', t, re.S):   # one entry at a time: newer objects carry checksum fields between ETag and Size, which made the old single pattern scan the whole page (CPU spin, stalled shards)
                k = re.search(r'<Key>(.*?)</Key>', blk, re.S); e = re.search(r'<ETag>&quot;(.*?)&quot;</ETag>', blk); z = re.search(r'<Size>(\d+)</Size>', blk)
                if k and e and z: keys.append((html.unescape(k.group(1)), e.group(1), int(z.group(1))))
            tk = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', t)
            if not tk: break
            tok = html.unescape(tk.group(1))
        total = sum(k[2] for k in keys)
        if total > 8e9:   # bigger than a runner can stage: record and skip (the S3 copy is larger than OpenNeuro's snapshot size)
            os.makedirs(stage + '/index', exist_ok=True)
            json.dump(dict(id=ds, name=s['name'], licence=s['licence'], files=len(keys), bytes=total, skipped_whole_dataset='larger than 8 GB, not copied', date=T), open(stage + '/index/%s.json' % ds, 'w'))
            pend.append(ds); print('dataset', ds, 'skipped (too big)', total, flush=True); continue
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
        staged += info['archive_size']
        if time.time() - last >= MIN_GAP or staged >= 5e9:
            if up is not None: await up
            d = stage; up = loop.run_in_executor(None, flush_rm, d, 'OpenNeuro CC0 datasets ' + ','.join(pend), 'datasets/openneuro'); last = time.time(); pend = []
            stage = W + '/stage%d' % int(time.time()); os.makedirs(stage, exist_ok=True); staged = 0
    if up is not None: await up
    if pend:
        wait = MIN_GAP - (time.time() - last)
        if wait > 0 and not DRY: await asyncio.sleep(wait)
        await loop.run_in_executor(None, flush, stage, 'OpenNeuro CC0 datasets ' + ','.join(pend), 'datasets/openneuro')

def bk_class_of_text(urls, texts):
    """Licence class of one NXML file from its licence elements and copyright statements (None when it has none)."""
    low = ' '.join(list(urls) + list(texts)).lower()
    if not low.strip(): return None
    nc = ('by-nc' in low) or ('noncommercial' in low) or ('non-commercial' in low) or ('non commercial' in low)
    sa = ('by-sa' in low) or ('sharealike' in low) or ('share-alike' in low) or ('share alike' in low)
    cc = ('creativecommons.org' in low) or ('creative commons' in low) or ('attribution license' in low) or ('cc by' in low) or ('cc0' in low)
    pd = ('public domain' in low or 'publicdomain' in low) and 'rights reserved' not in low
    if nc: return 'non-commercial'
    if sa: return 'share-alike'
    if cc or pd: return 'commercial-use' if cc else 'public-domain'
    return 'unclear'
BK_ORDER = ['public-domain', 'commercial-use', 'share-alike', 'non-commercial', 'unclear']   # least to most restrictive
def bk_scan(tp):
    """Read every NXML in a Bookshelf tarball; return (class, evidence, nxml count). Statements apply to all files of the book; the most restrictive chapter decides."""
    per = collections.Counter(); ev = collections.Counter(); n = 0
    with tarfile.open(tp, 'r:gz') as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith('.nxml')): continue
            n += 1
            if m.size > 30_000_000: continue
            x = tf.extractfile(m).read().decode('utf8', 'ignore')
            urls = set(); texts = []
            for blk in re.findall(r'<license\b.*?</license>|<ali:license_ref[^>]*>[^<]*</ali:license_ref>', x, re.S):
                for h in re.findall(r'(?:xlink:href|license-type)="([^"]*)"', blk): urls.add(h)
                t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', blk)).strip()
                if t: texts.append(t[:400])
            for cs in re.findall(r'<copyright-statement>(.*?)</copyright-statement>', x, re.S):
                t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', cs)).strip()
                if t: texts.append(t[:400])
            c = bk_class_of_text(urls, texts)
            if c: per[c] += 1; ev[' | '.join(sorted(urls))[:200] + ' || ' + (texts[0] if texts else '')[:200]] += 1
    if not per: return 'public-domain-no-statement', [], n
    worst = max(per, key=lambda c: BK_ORDER.index(c))
    return worst, [k for k, v in ev.most_common(4)], n
async def bookshelf(session, sem):
    """NCBI Bookshelf open-access books (pub/litarch tarballs): read each book's own licence statements, sort the tarball into a folder by licence class, copy all but the unclear ones."""
    import csv, io
    BASE = 'https://ftp.ncbi.nlm.nih.gov/pub/litarch/'
    async with session.get(BASE + 'file_list.csv') as resp: raw = (await resp.read()).decode('utf-8', 'replace')   # the list is not valid UTF-8 (a stray 0xAE)
    rows = list(csv.DictReader(io.StringIO(raw)))
    mine = [r for i, r in enumerate(rows) if i % SHARDS == SHARD and (not ONLY or r['Accession ID'] == ONLY)]
    print('bookshelf books for this shard:', len(mine), flush=True)
    have = set() if DRY else {f.split('/')[-1][:-5] for f in a.list_repo_files(R, repo_type='dataset') if f.startswith('datasets/bookshelf-oa/index/')}
    loop = asyncio.get_running_loop(); stage = W + '/stage'; staged = 0; pend = []; last = time.time() - MIN_GAP + MIN_GAP * SHARD / SHARDS; up = None
    os.makedirs(stage, exist_ok=True); tmpd = W + '/bk'; os.makedirs(tmpd, exist_ok=True)
    for r in mine:
        nbk = r['Accession ID']
        if nbk in have: continue
        url = BASE + r['File']; name = os.path.basename(r['File']); tp = tmpd + '/' + name; size = None
        for i in range(5):
            try:
                async with sem:
                    async with session.get(url) as resp:
                        if resp.status != 200: raise Exception('status %d' % resp.status)
                        size = int(resp.headers.get('Content-Length', 0))
                        if size > 9e9: break
                        with open(tp, 'wb') as f:
                            async for c in resp.content.iter_chunked(1 << 20): f.write(c)
                if os.path.getsize(tp) == size: break
                size = None
            except Exception as e: size = None
            await asyncio.sleep(min(2 + i * 3, 15))
        info = dict(accession=nbk, title=r['Title'], publisher=r['Publisher'], year=r['Publication Year'], file=r['File'], last_updated=r['Last Updated (YYYY-MM-DD HH:MM:SS)'], date=T)
        if size is None or size == 0: info.update(licence_class='download-failed', copied=False)
        elif size > 9e9: info.update(licence_class='not-scanned-too-big', bytes=size, copied=False)
        else:
            try: cls, ev, nx = await loop.run_in_executor(None, bk_scan, tp)
            except Exception as e: cls, ev, nx = 'scan-failed', [str(e)[:150]], 0
            info.update(licence_class=cls, evidence=ev, nxml_files=nx, bytes=size, copied=cls in ('public-domain', 'public-domain-no-statement', 'commercial-use', 'share-alike', 'non-commercial'))
            if info['copied']:
                dst = stage + '/' + cls + '/' + name; os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(tp, dst)
                info.update(sha256=sha(dst), path='datasets/bookshelf-oa/%s/%s' % (cls, name)); staged += size
        if os.path.exists(tp): os.remove(tp)
        os.makedirs(stage + '/index', exist_ok=True); json.dump(info, open(stage + '/index/%s.json' % nbk, 'w'))
        pend.append(nbk); print('book', nbk, info.get('licence_class'), size, flush=True)
        if time.time() - last >= MIN_GAP or staged >= 5e9:
            if up is not None: await up
            d = stage; up = loop.run_in_executor(None, flush_rm, d, 'Bookshelf open-access books %d (%s ...)' % (len(pend), pend[0]), 'datasets/bookshelf-oa'); last = time.time(); pend = []
            stage = W + '/stage%d' % int(time.time()); os.makedirs(stage, exist_ok=True); staged = 0
    if up is not None: await up
    if pend:
        wait = MIN_GAP - (time.time() - last)
        if wait > 0 and not DRY: await asyncio.sleep(wait)
        await loop.run_in_executor(None, flush, stage, 'Bookshelf open-access books %d (%s ...)' % (len(pend), pend[0]), 'datasets/bookshelf-oa')
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

async def orcid(session, sem):
    """ORCID Public Data File 2026, summaries only (CC0), stored in 8 GB parts: shard n copies part n using 8 parallel range connections.
    The whole-file md5 from Figshare is checked afterwards by streaming the parts back from Hugging Face."""
    async with session.get('https://api.figshare.com/v2/articles/34697766') as r: d = await r.json()
    assert d['license']['name'] == 'CC0'
    f = [x for x in d['files'] if 'summaries' in x['name']][0]; size = f['size']; PART = 8_000_000_000; n = SHARD
    nparts = -(-size // PART)
    if n >= nparts: print('no part for this shard', flush=True); return
    dest = 'datasets/orcid/%s.part%02d' % (f['name'], n)
    if not DRY and a.file_exists(R, dest, repo_type='dataset'): print('skip', dest, flush=True); return
    lo = n * PART; hi = min(size, lo + PART) - 1; path = W + '/part'; os.makedirs(W, exist_ok=True)
    with open(path, 'wb') as fh: fh.truncate(hi - lo + 1)
    sub = -(-(hi - lo + 1) // 8)
    async def piece(i):
        a0 = lo + i * sub; b0 = min(hi, a0 + sub - 1); pos = a0
        for t in range(8):
            try:
                async with sem:
                    async with session.get(f['download_url'], headers={'Range': 'bytes=%d-%d' % (pos, b0)}) as r:
                        if r.status not in (200, 206): raise Exception('status %d' % r.status)
                        with open(path, 'r+b') as fh:
                            fh.seek(pos - lo)
                            async for c in r.content.iter_chunked(1 << 22): fh.write(c); pos += len(c)
                if pos == b0 + 1: return
            except Exception: await asyncio.sleep(3)
        raise Exception('piece failed %d' % i)
    await asyncio.gather(*[piece(i) for i in range(8) if lo + i * sub <= hi])
    assert os.path.getsize(path) == hi - lo + 1
    info = dict(part=n, file=f['name'], range='%d-%d' % (lo, hi), size=hi - lo + 1, sha256=sha(path), whole_file_md5_from_figshare=f['computed_md5'], date=T)
    json.dump(info, open(W + '/part.json', 'w'))
    if DRY: print('DRY would upload', dest, info, flush=True); return
    a.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=R, repo_type='dataset', commit_message='ORCID 2026 summaries part %d (shard %d)' % (n, SHARD))
    a.upload_file(path_or_fileobj=W + '/part.json', path_in_repo='datasets/orcid/_parts/part_%02d.json' % n, repo_id=R, repo_type='dataset', commit_message='ORCID part %d index' % n)
    print('uploaded part', n, flush=True)
async def main():
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    conn = aiohttp.TCPConnector(limit=CONC, limit_per_host=CONC, ttl_dns_cache=300); sem = asyncio.Semaphore(CONC)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120), headers={'User-Agent': 'moby-hf-copy'}, trust_env=True) as session:
        await {'openneuro': openneuro, 'vhp': vhp, 'orcid': orcid, 'bookshelf': bookshelf}[JOB](session, sem)
    print('SHARD DONE', JOB, SHARD, flush=True)
asyncio.run(main())
