# download raw nhanes data (.xpt) and codebook (.htm) files from cdc
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# each 2-year cycle has a letter (G = 2011-2012, H = 2013-2014, etc.)
CYCLES = [(2011, 'G'), (2013, 'H'), (2015, 'I'), (2017, 'J')]
MODULES = 'DEMO GHB DIQ BMX BPX PAQ SMQ ALQ DR1TOT SLQ WHQ MCQ'.split()


# simple html parser to collect all links on the page
class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def retrieve(url):
    last = None
    # try 3 times in case connection fails
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Type0 academic data preparation'})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read(), response.url
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'{url}: {last}')


def download(task):
    year, module, stem, data_url, doc_url = task
    folder = ROOT / 'raw' / f'{year}-{year+1}'
    folder.mkdir(parents=True, exist_ok=True)
    entry = {'cycle': f'{year}-{year+1}', 'module': module, 'stem': stem}
    for kind, url, ext in [('data', data_url, '.xpt'), ('codebook', doc_url, '.htm')]:
        dest = folder / (stem + ext)
        if dest.exists():
            blob = dest.read_bytes()
            final_url = url
        else:
            blob, final_url = retrieve(url)
            # check that we really got an xpt or html file, not an error page
            if kind == 'data' and not blob.startswith(b'HEADER RECORD'):
                raise ValueError(f'XPT yerine beklenmeyen içerik: {url}')
            if kind == 'codebook' and b'<html' not in blob[:1000].lower():
                raise ValueError(f'Kod kitabı HTML değil: {url}')
            # write to .part first then rename when finished
            temp = dest.with_suffix(ext + '.part')
            temp.write_bytes(blob)
            temp.replace(dest)
        entry[kind] = {'url': url, 'resolved_url': final_url, 'path': str(dest.relative_to(ROOT)),
                       'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
    print(f"Hazır: {stem}", flush=True)
    return entry


def main(modules=MODULES, restricted_years=None):
    restricted_years = restricted_years or {}
    tasks = []
    for year, suffix in CYCLES:
        index_url = f'https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Cycle={year}-{year+1}'
        page, _ = retrieve(index_url)
        folder = ROOT / 'raw' / f'{year}-{year+1}'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'index.htm').write_bytes(page)
        # get file links from the cdc cycle page
        parser = Links()
        parser.feed(page.decode('utf-8-sig'))
        links = {Path(urllib.parse.urlparse(h).path).name.lower(): urllib.parse.urljoin(index_url, h)
                 for h in parser.links}
        for module in modules:
            if module in restricted_years and year not in restricted_years[module]:
                continue
            stem = f'{module}_{suffix}'
            data_url = links.get(stem.lower() + '.xpt')
            doc_url = links.get(stem.lower() + '.htm')
            if not data_url or not doc_url:
                raise ValueError(f'Resmî listede bağlantı eksik: {stem}')
            tasks.append((year, module, stem, data_url, doc_url))
    entries, errors = [], []
    # download 6 files at once to make it faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download, task): task[2] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                entries.append(future.result())
            except Exception as exc:
                errors.append({'stem': futures[future], 'error': str(exc)})
                print(f'HATA: {futures[future]}: {exc}', flush=True)
    manifest = {'retrieved_on': date.today().isoformat(), 'files': sorted(entries, key=lambda x: x['stem']), 'errors': errors}
    (ROOT / 'source-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Tamamlanan: {len(entries)}/{len(tasks)}; hata: {len(errors)}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
