"""OpenAlex open-access gap analysis (one shard of SHARDS).
Reads this shard's part files of the OpenAlex 'works' snapshot from our public Hugging Face copy, keeps health-sciences works (primary topic domain 4), and counts them by open-access status, licence class and host; writes the commercial-use and non-commercial Creative Commons works that are NOT in PubMed Central (no PMCID, no PMC or Europe PMC link) to a candidate list (id, doi, licence, links, host) for a later, licence-checked fetch.
Env: SHARD, SHARDS, HF_TOKEN (read and write), DRY=1 (no upload), LIMIT=n (only the first n files of this shard, for tests)."""
import os, re, json, gzip, time, shutil, collections, multiprocessing as mp
from huggingface_hub import HfApi, hf_hub_download
R = 'Middle-child/moby-public-assets'; SHARD = int(os.environ['SHARD']); SHARDS = int(os.environ['SHARDS']); DRY = os.environ.get('DRY') == '1'
LIMIT = int(os.environ.get('LIMIT', '0')); W = os.environ.get('RUNNER_TEMP', '/tmp') + '/oagap%d' % SHARD
COMM = {'cc-by', 'cc-by-sa', 'cc-by-nd', 'cc0', 'public-domain'}
RX = re.compile(r'"license":"[^"]')
def klass(lic):
    if not lic: return 'no-licence-stated'
    if lic in COMM: return 'cc-commercial'
    if lic.startswith('cc-by-nc'): return 'cc-noncommercial'
    return 'other-licence'
def one(path):
    t0 = time.time(); local = hf_hub_download(R, path, repo_type='dataset', local_dir=W + '/dl', token=os.environ.get('HF_TOKEN'))
    c = collections.Counter(); hosts = collections.Counter(); out = []
    with gzip.open(local, 'rt', encoding='utf8') as g:
        for line in g:
            c['lines'] += 1
            if 'openalex.org/domains/4"' not in line: continue
            try: r = json.loads(line)
            except Exception: c['bad_json'] += 1; continue
            pt = r.get('primary_topic') or {}
            if ((pt.get('domain') or {}).get('id') or '').rsplit('/', 1)[-1] != '4': continue
            c['health'] += 1; c['health_type_' + str(r.get('type'))] += 1
            oa = r.get('open_access') or {}
            if not oa.get('is_oa'): c['health_closed'] += 1; continue
            c['health_oa'] += 1; c['oa_status_' + str(oa.get('oa_status'))] += 1
            b = r.get('best_oa_location') or {}
            lic = b.get('license'); k = klass(lic); c['licclass_' + k] += 1; c['lic_' + str(lic)] += 1
            urls = ' '.join(str(x or '') for x in (b.get('landing_page_url'), b.get('pdf_url'), (r.get('ids') or {}).get('pmcid')))
            in_pmc = bool((r.get('ids') or {}).get('pmcid')) or 'ncbi.nlm.nih.gov/pmc' in urls or 'europepmc.org' in urls
            c['in_pmc' if in_pmc else 'not_in_pmc'] += 1
            if k in ('cc-commercial', 'cc-noncommercial'):
                c[('cand_' if not in_pmc else 'inpmc_') + k] += 1
                s = b.get('source') or {}
                host = s.get('host_organization_name') or s.get('display_name') or 'unknown'
                if not in_pmc:
                    hosts[(k, host, s.get('type'))] += 1
                    out.append(dict(id=r.get('id'), doi=r.get('doi'), year=r.get('publication_year'), type=r.get('type'), licence=lic, oa_status=oa.get('oa_status'), pdf=b.get('pdf_url'), landing=b.get('landing_page_url'), host=host, source=s.get('display_name'), source_type=s.get('type'), version=b.get('version')))
    os.remove(local)
    return path, c, hosts, out, time.time() - t0
if __name__ == '__main__':
    a = HfApi(token=os.environ.get('HF_TOKEN'))
    files = sorted(e.path for e in a.list_repo_tree(R, path_in_repo='datasets/openalex/data/jsonl/works', repo_type='dataset', recursive=True) if e.__class__.__name__ == 'RepoFile' and e.path.endswith('.gz'))
    mine = [f for i, f in enumerate(files) if i % SHARDS == SHARD]
    if LIMIT: mine = mine[:LIMIT]
    print('works files total', len(files), 'this shard', len(mine), flush=True)
    os.makedirs(W, exist_ok=True); tot = collections.Counter(); hosts = collections.Counter(); n = 0
    cand = gzip.open(W + '/candidates_shard%02d.jsonl.gz' % SHARD, 'wt', encoding='utf8')
    with mp.Pool(3) as pool:
        for path, c, h, out, dt in pool.imap_unordered(one, mine):
            tot.update(c); hosts.update(h); n += 1
            for r in out: cand.write(json.dumps(r) + '\n')
            print('%d/%d %s %.0fs health=%d cand=%d' % (n, len(mine), path.split('/')[-2], dt, c['health'], c['cand_cc-commercial'] + c['cand_cc-noncommercial']), flush=True)
    cand.close()
    top = [dict(class_=k[0], host=k[1], source_type=k[2], works=v) for k, v in hosts.most_common(5000)]
    json.dump(dict(shard=SHARD, shards=SHARDS, files=len(mine), counts=dict(tot), hosts_top5000=top, hosts_total=len(hosts), date=time.strftime('%Y-%m-%d')), open(W + '/summary_shard%02d.json' % SHARD, 'w'))
    print('SHARD DONE', dict(tot), flush=True)
    if not DRY:
        stage = W + '/up'; os.makedirs(stage, exist_ok=True)
        for f in ('candidates_shard%02d.jsonl.gz' % SHARD, 'summary_shard%02d.json' % SHARD): shutil.copy(W + '/' + f, stage + '/' + f)
        for t in range(8):
            try: a.upload_folder(repo_id=R, repo_type='dataset', folder_path=stage, path_in_repo='datasets/openalex-oa-gap', commit_message='OpenAlex open-access gap analysis, shard %d of %d' % (SHARD, SHARDS)); print('uploaded', flush=True); break
            except Exception as e: print('upload retry', str(e)[:100], flush=True); time.sleep(60 + 30 * SHARD)
