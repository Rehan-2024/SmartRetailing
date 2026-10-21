"""Verify supplied files against bytes downloaded directly from UCI.

Unknown-provenance files are inventoried, never trained on. Raw originals stay local.
"""
import argparse, hashlib, io, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

SOURCES = [
    ('retail', 502, 'online+retail+ii', 'transactions/online_retail_II.xlsx', 'online_retail_II.xlsx', 'Chen, D. (2012)', '10.24432/C5CG6D'),
    ('purchase', 468, 'online+shoppers+purchasing+intention+dataset', 'digital/online_shoppers_intention.csv', 'online_shoppers_intention.csv', 'Sakar, C. & Kastro, Y. (2018)', '10.24432/C5F88Q'),
    ('wholesale', 292, 'wholesale+customers', 'Wholesale customers data.csv', 'Wholesale customers data.csv', 'Cardoso, M. (2013)', '10.24432/C5030X'),
]
EXCLUDED = ['marketing_campaign.csv', 'store/store_sales_data (2).csv', 'geo/geolocations-indian-cities.csv', 'geo/Pincode_Dataset.csv']
sha = lambda data: hashlib.sha256(data).hexdigest()

def verify(root, output, cache):
    cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, identifier, slug, relative, member, author, doi in SOURCES:
        url = f'https://archive.ics.uci.edu/static/public/{identifier}/{slug}.zip'
        archive = cache / f'uci-{identifier}.zip'
        if not archive.exists():
            print(f'Downloading official UCI {identifier}...', flush=True)
            with urlopen(url, timeout=180) as response:
                archive.write_bytes(response.read())
        raw = archive.read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            official = bundle.read(member)
        local = (root / relative).read_bytes()
        match = sha(local) == sha(official)
        entries.append(dict(source=name, file=relative, status='verified-exact' if match else 'REJECTED-mismatch',
            sha256=sha(local), officialMemberSHA256=sha(official), archiveSHA256=sha(raw), bytes=len(local),
            archiveURL=url, sourceURL=f'https://archive.ics.uci.edu/dataset/{identifier}/{slug}',
            archiveMember=member, license='CC BY 4.0', attribution=author, doi=doi))
        print(name, entries[-1]['status'], flush=True)
    for relative in EXCLUDED:
        path = root / relative
        entries.append(dict(file=relative, status='excluded-unverified', exists=path.exists(),
            sha256=sha(path.read_bytes()) if path.exists() else None,
            reason='Real-world origin and redistribution license of this exact file are unverified. Excluded from training, descriptive outputs and notebook inputs; this does not prove it is synthetic.'))
    report = dict(checkedAt=datetime.now(timezone.utc).isoformat(), policy='Only exact official UCI copies are accepted by the public benchmark pipelines. No generated or augmented training rows.', sources=entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf8')
    if any(r['status'].startswith('REJECTED') for r in entries):
        raise ValueError('Source mismatch: see audit; do not train until resolved.')
    return report

def require_verified(root, names):
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'public/data/sources.json').read_text())
    for name in names:
        row = next(r for r in manifest['sources'] if r.get('source') == name)
        if row['status'] != 'verified-exact' or sha((root / row['file']).read_bytes()) != row['officialMemberSHA256']:
            raise ValueError(f'{name}: file is not an audited exact official UCI copy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('..'))
    parser.add_argument('--output', type=Path, default=Path('public/data/sources.json'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/provenance'))
    args = parser.parse_args()
    verify(args.data_root, args.output, args.cache)
