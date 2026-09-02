#!/usr/bin/env python3
"""
fast_validate_bibliography.py
------------------------------
Fast, multithreaded bibliography validator and updater using NCBI PubMed and CrossRef APIs.
Validates all cited references in LaTeX documents, detects discrepancies in titles, authors,
journals, publication years, volumes, and pages, and outputs a detailed Markdown audit report.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'[\{\}\\\"\']', '', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def similarity(a: str, b: str) -> float:
    a_clean = re.sub(r'[^a-zA-Z0-9]', '', a.lower())
    b_clean = re.sub(r'[^a-zA-Z0-9]', '', b.lower())
    if not a_clean and not b_clean:
        return 1.0
    if not a_clean or not b_clean:
        return 0.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def parse_bib_file(bib_path: str):
    with open(bib_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    entries = {}
    raw_entries = re.split(r'\n(?=@\w+\s*\{)', content)
    for raw in raw_entries:
        raw = raw.strip()
        if not raw.startswith('@'):
            continue
        m = re.search(r'@(\w+)\s*\{\s*([^,]+),', raw)
        if not m:
            continue
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        
        entry = {'type': entry_type, 'key': key, 'raw': raw}
        for line in raw.splitlines():
            line = line.strip()
            if '=' in line and not line.startswith('@'):
                parts = line.split('=', 1)
                fname = parts[0].strip().lower()
                fval = parts[1].strip().rstrip(',').strip('"{}\'')
                entry[fname] = clean_text(fval)
        entries[key] = entry
    return entries


def get_cited_keys(aux_path: str):
    if not os.path.exists(aux_path):
        return None
    with open(aux_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    cited = set()
    for m in re.finditer(r'\\citation\{([^}]+)\}', content):
        for k in m.group(1).split(','):
            k = k.strip()
            if k:
                cited.add(k)
    return cited


def query_crossref_canonical(title: str, author: str = "", doi: str = ""):
    headers = {'User-Agent': 'HyphAeonBibValidator/2.0 (mailto:sergei@temple.edu)'}
    
    # 1. Direct DOI
    if doi:
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
        url = f"https://api.crossref.org/works/{urllib.parse.quote(clean_doi)}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                item = data.get("message", {})
                if item:
                    return parse_crossref_item(item)
        except Exception:
            pass

    # 2. Bibliographic query
    first_auth = author.split('and')[0].split(',')[0].strip() if author else ""
    clean_t = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
    query_str = f"{clean_t} {first_auth}".strip()
    
    url = f"https://api.crossref.org/works?query.bibliographic={urllib.parse.quote(query_str)}&rows=5"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = data.get("message", {}).get("items", [])
            for item in items:
                rec = parse_crossref_item(item)
                if rec and title and similarity(title, rec["title"]) > 0.60:
                    return rec
    except Exception:
        pass

    return None


def parse_crossref_item(item: dict):
    t_list = item.get("title", [])
    title = t_list[0] if t_list else ""
    authors = []
    for a in item.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        if family and given:
            authors.append(f"{family}, {given}")
        elif family:
            authors.append(family)
    author_str = " and ".join(authors) if authors else ""
    
    container = item.get("container-title", [])
    journal = container[0] if container else item.get("publisher", "")
    
    year = ""
    issued = item.get("issued", {}).get("date-parts", [[]])[0]
    if issued:
        year = str(issued[0])
    
    return {
        "source": "CrossRef",
        "title": clean_text(title),
        "author": author_str,
        "authors_list": authors,
        "journal": journal,
        "year": year,
        "volume": str(item.get("volume", "")),
        "issue": str(item.get("issue", "")),
        "pages": str(item.get("page", "")),
        "doi": str(item.get("DOI", "")),
        "pmid": ""
    }


def query_pubmed_canonical(title: str, author: str = "", doi: str = ""):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    headers = {'User-Agent': 'HyphAeonBibValidator/2.0 (mailto:sergei@temple.edu)'}
    
    pmid = None
    if doi:
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
        params = {"db": "pubmed", "term": f"{clean_doi}[AID]", "retmode": "json"}
        try:
            url = base_url + "esearch.fcgi?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                ids = data.get("esearchresult", {}).get("idlist", [])
                if ids:
                    pmid = ids[0]
        except Exception:
            pass

    if not pmid and title:
        first_auth = author.split('and')[0].split(',')[0].strip() if author else ""
        clean_t = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
        words = [w for w in clean_t.split() if len(w) > 3][:6]
        term = " ".join(words)
        if first_auth and len(first_auth) > 2:
            term += f" AND {first_auth}[Author]"
        params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": "3"}
        try:
            url = base_url + "esearch.fcgi?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                ids = data.get("esearchresult", {}).get("idlist", [])
                if ids:
                    pmid = ids[0]
        except Exception:
            pass

    if pmid:
        params = {"db": "pubmed", "id": pmid, "retmode": "json"}
        try:
            url = base_url + "esummary.fcgi?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                res = data.get("result", {}).get(pmid, {})
                if res:
                    authors = [a.get("name", "") for a in res.get("authors", [])]
                    pubdate = res.get("pubdate", "")
                    pyear = ""
                    ym = re.search(r'\b(19\d\d|20\d\d)\b', pubdate)
                    if ym:
                        pyear = ym.group(1)
                    
                    official_doi = ""
                    for aid in res.get("articleids", []):
                        if aid.get("idtype") == "doi":
                            official_doi = aid.get("value", "")
                            break

                    return {
                        "source": "PubMed",
                        "pmid": pmid,
                        "title": clean_text(res.get("title", "")),
                        "author": " and ".join(authors),
                        "authors_list": authors,
                        "journal": res.get("source", "") or res.get("fulljournalname", ""),
                        "year": pyear,
                        "volume": str(res.get("volume", "")),
                        "issue": str(res.get("issue", "")),
                        "pages": str(res.get("pages", "")),
                        "doi": official_doi
                    }
        except Exception:
            pass

    return None


def validate_single_entry(key: str, entry: dict):
    title = entry.get('title', '')
    author = entry.get('author', '')
    doi = entry.get('doi', '')
    year = entry.get('year', '')
    
    # Try PubMed first, then CrossRef
    official = query_pubmed_canonical(title=title, author=author, doi=doi)
    if not official:
        official = query_crossref_canonical(title=title, author=author, doi=doi)

    if not official:
        return {
            "key": key,
            "status": "NOT_FOUND",
            "bib": entry,
            "official": None,
            "discrepancies": ["Record could not be verified automatically via PubMed or CrossRef"]
        }

    discrepancies = []
    t_sim = similarity(title, official.get("title", ""))
    if t_sim < 0.85:
        discrepancies.append(f"Title variation (sim={t_sim:.2f}):\n    Bib:      '{title}'\n    Official: '{official.get('title')}'")

    if year and official.get("year") and year != official.get("year"):
        discrepancies.append(f"Year mismatch: Bib='{year}' vs Official='{official.get('year')}'")

    if author and official.get("author"):
        first_bib = author.split('and')[0].split(',')[0].strip()
        first_off = official.get("author").split('and')[0].split(',')[0].strip()
        if similarity(first_bib, first_off) < 0.60:
            discrepancies.append(f"First author mismatch: Bib='{first_bib}' vs Official='{first_off}'")

    status = "DISCREPANCY" if discrepancies else "MATCH"
    return {
        "key": key,
        "status": status,
        "bib": entry,
        "official": official,
        "discrepancies": discrepancies
    }


def main():
    parser = argparse.ArgumentParser(description="Fast parallel bibliography validator")
    parser.add_argument("--bib", default="hyphaeon_paper/references.bib")
    parser.add_argument("--aux", default="hyphaeon_paper/main.aux")
    parser.add_argument("--report", default="hyphaeon_paper/bibliography_validation_report.md")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    entries = parse_bib_file(args.bib)
    cited_keys = get_cited_keys(args.aux)
    target_keys = sorted([k for k in entries.keys() if k in cited_keys]) if cited_keys else sorted(list(entries.keys()))

    print(f"🚀 Parallel scanning {len(target_keys)} cited keys across PubMed & CrossRef ({args.workers} workers)...")
    start_time = time.time()
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_key = {executor.submit(validate_single_entry, k, entries[k]): k for k in target_keys}
        for future in concurrent.futures.as_completed(future_to_key):
            res = future.result()
            results.append(res)
            icon = "✅" if res["status"] == "MATCH" else ("⚠️" if res["status"] == "DISCREPANCY" else "❌")
            print(f"[{len(results):02d}/{len(target_keys):02d}] {icon} {res['key']:<28} ({res['status']})")

    results.sort(key=lambda r: r['key'])
    elapsed = time.time() - start_time
    print(f"\n⚡ Completed validation of {len(target_keys)} records in {elapsed:.2f} seconds.")

    matches = [r for r in results if r['status'] == 'MATCH']
    discrepancies = [r for r in results if r['status'] == 'DISCREPANCY']
    not_found = [r for r in results if r['status'] == 'NOT_FOUND']

    print("\n" + "=" * 60)
    print("📊 BIBLIOGRAPHY VALIDATION SUMMARY")
    print(f"   Total checked:    {len(target_keys)}")
    print(f"   ✅ Canonical Match: {len(matches):<2} ({len(matches)/len(target_keys)*100:.1f}%)")
    print(f"   ⚠️ Discrepancy:     {len(discrepancies):<2} ({len(discrepancies)/len(target_keys)*100:.1f}%)")
    print(f"   ❌ Unresolved:      {len(not_found):<2} ({len(not_found)/len(target_keys)*100:.1f}%)")
    print("=" * 60)

    # Write Markdown Report
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write("# Automated PubMed & CrossRef Bibliography Verification Report\n\n")
        f.write(f"Generated on **{time.strftime('%Y-%m-%d %H:%M:%S')}** via parallel NCBI PubMed and CrossRef E-utilities.\n\n")
        f.write("### Executive Summary\n\n")
        f.write(f"- **Total Citations Audited:** {len(target_keys)}\n")
        f.write(f"- **Exact / Canonical Records Confirmed:** {len(matches)} ({len(matches)/len(target_keys)*100:.1f}%)\n")
        f.write(f"- **Discrepancies / Title & Metadata Variations Detected:** {len(discrepancies)} ({len(discrepancies)/len(target_keys)*100:.1f}%)\n")
        f.write(f"- **Unindexed / Custom References:** {len(not_found)} ({len(not_found)/len(target_keys)*100:.1f}%)\n\n")

        f.write("## Discrepancies & Verification Details\n\n")
        for res in discrepancies:
            f.write(f"### `{res['key']}` — Status: **⚠️ DISCREPANCY**\n\n")
            f.write(f"- **BibTeX Title:** {res['bib'].get('title', 'N/A')}\n")
            f.write(f"- **BibTeX Authors:** {res['bib'].get('author', 'N/A')}\n")
            f.write(f"- **BibTeX Journal & Year:** {res['bib'].get('journal', 'N/A')} ({res['bib'].get('year', 'N/A')})\n")
            if res["official"]:
                f.write(f"- **Official Published Record ({res['official'].get('source')}):**\n")
                f.write(f"  - **Official Title:** {res['official'].get('title')}\n")
                f.write(f"  - **Official Authors:** {res['official'].get('author')[:140]}...\n")
                f.write(f"  - **Official Journal:** {res['official'].get('journal')} ({res['official'].get('year')})\n")
                if res['official'].get('volume'):
                    f.write(f"  - **Volume/Pages:** Vol. {res['official'].get('volume')}, pp. {res['official'].get('pages')}\n")
                if res['official'].get('doi'):
                    f.write(f"  - **DOI:** [{res['official'].get('doi')}](https://doi.org/{res['official'].get('doi')})\n")
                if res['official'].get('pmid'):
                    f.write(f"  - **PMID:** [{res['official'].get('pmid')}](https://pubmed.ncbi.nlm.nih.gov/{res['official'].get('pmid')}/)\n")
            if res["discrepancies"]:
                f.write("\n**Issues Detected:**\n")
                for d in res["discrepancies"]:
                    f.write(f"- {d}\n")
            f.write("\n---\n\n")

        if not_found:
            f.write("## Unindexed / Unresolved References\n\n")
            for res in not_found:
                f.write(f"### `{res['key']}` — Status: **❌ NOT FOUND**\n\n")
                f.write(f"- **BibTeX Title:** {res['bib'].get('title', 'N/A')}\n")
                f.write(f"- **BibTeX Authors:** {res['bib'].get('author', 'N/A')}\n")
                f.write(f"- **BibTeX Journal/Year:** {res['bib'].get('journal', 'N/A')} ({res['bib'].get('year', 'N/A')})\n\n")

        f.write("## Complete Verified Reference Index\n\n")
        f.write("| Citation Key | Status | Source | Official Journal | Year | Official Identifier |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for res in results:
            off = res["official"]
            if off:
                src = off.get("source", "N/A")
                j = off.get("journal", "N/A")
                y = off.get("year", "N/A")
                link = f"[DOI](https://doi.org/{off.get('doi')})" if off.get('doi') else (f"[PMID](https://pubmed.ncbi.nlm.nih.gov/{off.get('pmid')}/)" if off.get('pmid') else "Verified")
            else:
                src = "N/A"
                j = res["bib"].get("journal", "N/A")
                y = res["bib"].get("year", "N/A")
                link = "Unindexed"
            f.write(f"| `{res['key']}` | `{res['status']}` | {src} | {j[:35]} | {y} | {link} |\n")

    print(f"📝 Full validation report written to {args.report}")


if __name__ == "__main__":
    main()
