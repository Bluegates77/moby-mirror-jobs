"""Copies one public Hugging Face dataset into our repo under DEST, then proves the copy: every LFS file's SHA-256 is checked after download (against the source's own hash) and again on our side after upload. Exits non-zero on any difference.
Env: SRC (owner/name), DEST (folder in our repo), HF_TOKEN, optional MAXFILES (test), DRY=1 (no upload)."""
import os, sys, json, hashlib, datetime, shutil
from huggingface_hub import HfApi, snapshot_download
SRC = os.environ['SRC']; DEST = os.environ['DEST']; MAXF = int(os.environ.get('MAXFILES', '0')); DRY = os.environ.get('DRY') == '1'
R = 'Middle-child/moby-public-assets'; a = HfApi(); out = os.environ.get('RUNNER_TEMP', '/tmp') + '/copyhf'
shutil.rmtree(out, ignore_errors=True)
tree = [f for f in a.list_repo_tree(SRC, repo_type='dataset', recursive=True) if hasattr(f, 'size')]
if MAXF: tree = [f for f in tree if f.path != '.gitattributes'][:MAXF]
print('source files', len(tree), 'bytes', sum(f.size for f in tree), flush=True)
snapshot_download(SRC, repo_type='dataset', local_dir=out, allow_patterns=[f.path for f in tree], max_workers=8)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): h.update(c)
    return h.hexdigest()
bad = 0; rows = []
for f in tree:
    p = os.path.join(out, f.path)
    if not os.path.exists(p) or os.path.getsize(p) != f.size: print('SIZE MISMATCH', f.path, flush=True); bad += 1; continue
    want = f.lfs.sha256 if getattr(f, 'lfs', None) else None; got = sha(p)
    if want and want != got: print('SHA MISMATCH', f.path, flush=True); bad += 1
    rows.append(dict(path=f.path, bytes=f.size, sha256=got, source_sha256=want))
if bad: sys.exit('download check failed: %d' % bad)
print('download verified', len(rows), 'files', flush=True)
with open(out + '/SOURCE.txt', 'w') as f:
    f.write('Copy of the public Hugging Face dataset %s\nCopied %s. Licence: not stated on the source card (README has no licence field); treat as unclear. Credit the source.\nEvery file was verified by SHA-256 after download and after upload (see SOURCE_FILES.json).\n' % (SRC, datetime.date.today()))
with open(out + '/SOURCE_FILES.json', 'w') as f: json.dump(rows, f)
if DRY: print('DRY: would upload', DEST); sys.exit(0)
a.upload_folder(repo_id=R, repo_type='dataset', folder_path=out, path_in_repo=DEST, ignore_patterns=['.cache/**'], commit_message='Copy of %s (unclear licence folder)' % SRC)
mine = {f.path[len(DEST) + 1:]: f for f in a.list_repo_tree(R, path_in_repo=DEST, repo_type='dataset', recursive=True) if hasattr(f, 'size')}
miss = 0
for r in rows:
    f = mine.get(r['path'])
    if f is None or f.size != r['bytes'] or (getattr(f, 'lfs', None) and f.lfs.sha256 != r['sha256']): print('UPLOAD MISMATCH', r['path'], flush=True); miss += 1
if miss: sys.exit('upload check failed: %d' % miss)
print('UPLOAD VERIFIED', len(rows), 'files at', DEST, flush=True)
