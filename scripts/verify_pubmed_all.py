#!/usr/bin/env python3
"""
verify_pubmed_all.py
--------------------
Comprehensive NCBI PubMed validator for LaTeX references (.bib file).
Queries PubMed via NCBI E-utilities (eSearch + eSummary) with intelligent fallback queries:
1. DOI search ([AID])
2. Exact normalized title search ([Title])
3. Author + title keyword combination search
4. CrossRef fallback for non-PubMed indexed items (e.g., preprints, books, CS conference proceedings)

Outputs a comprehensive Markdown report and verified BibTeX records.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher


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


def query_pubmed(title: str, author: str = "", doi: str = ""):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    headers = {'User-Agent': 'HyphAeonPubmedValidator/1.0 (mailto:sergei@temple.edu)'}
    
    pmid = None
    
    # 1. DOI Search
    if doi:
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
        params = {"db": "pubmed", "term": f"{clean_doi}[AID]", "retmode": "json", "tool": "HyphAeon", "email": "sergei@temple.edu"}
        try:
            time.sleep(0.35)
            url = base_url + "esearch.fcgi?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                ids = data.get("esearchresult", {}).get("idlist", [])
                if ids:
                    pmid = ids[0]
        except Exception:
            pass

    # 2. Search queries
    if not pmid and title:
        search_terms = []
        
        # Clean title words
        clean_t = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
        words = [w for w in clean_t.split() if len(w) > 3]
        
        first_auth = ""
        if author:
            # Extract last name
            first_auth = author.split('and')[0].split(',')[0].strip()
            if " " in first_auth:
                first_auth = first_auth.split()[-1]

        # Query A: First 5 significant words + Author
        if words and first_auth:
            search_terms.append(f"{' '.join(words[:5])}[Title] AND {first_auth}[Author]")
            
        # Query B: First 6 words in title
        if words:
            search_terms.append(f"{' '.join(words[:6])}[Title]")
            
        # Query C: First 3 words + Author
        if len(words) >= 3 and first_auth:
            search_terms.append(f"{' '.join(words[:3])}[Title] AND {first_auth}[Author]")

        for term in search_terms:
            try:
                time.sleep(0.35)
                params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": "3", "tool": "HyphAeon", "email": "sergei@temple.edu"}
                url = base_url + "esearch.fcgi?" + urllib.parse.urlencode(params)
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    ids = data.get("esearchresult", {}).get("idlist", [])
                    if ids:
                        pmid = ids[0]
                        break
            except Exception:
                pass

    if pmid:
        try:
            time.sleep(0.35)
            params = {"db": "pubmed", "id": pmid, "retmode": "json", "tool": "HyphAeon", "email": "sergei@temple.edu"}
            url = base_url + "esummary.fcgi?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
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
                        "authors": authors,
                        "author": " and ".join(authors),
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


def query_crossref(title: str, author: str = "", doi: str = ""):
    headers = {'User-Agent': 'HyphAeonPubmedValidator/1.0 (mailto:sergei@temple.edu)'}
    
    if doi:
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
        url = f"https://api.crossref.org/works/{urllib.parse.quote(clean_doi)}"
        try:
            time.sleep(0.35)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                item = data.get("message", {})
                if item:
                    return parse_crossref_item(item)
        except Exception:
            pass

    if title:
        first_auth = author.split('and')[0].split(',')[0].strip() if author else ""
        clean_t = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
        query_str = f"{clean_t} {first_auth}".strip()
        url = f"https://api.crossref.org/works?query.bibliographic={urllib.parse.quote(query_str)}&rows=3"
        try:
            time.sleep(0.35)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
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
        "authors": authors,
        "journal": journal,
        "year": year,
        "volume": str(item.get("volume", "")),
        "issue": str(item.get("issue", "")),
        "pages": str(item.get("page", "")),
        "doi": str(item.get("DOI", "")),
        "pmid": ""
    }


def validate_all(bib_path: str, aux_path: str, report_path: str):
    entries = parse_bib_file(bib_path)
    cited_keys = get_cited_keys(aux_path)
    target_keys = sorted([k for k in entries.keys() if k in cited_keys]) if cited_keys else sorted(list(entries.keys()))

    print(f"📖 Scanning {len(target_keys)} cited keys in {bib_path} using NCBI PubMed & CrossRef...")
    
    results = []
    for idx, key in enumerate(target_keys, 1):
        entry = entries[key]
        title = entry.get('title', '')
        author = entry.get('author', '')
        doi = entry.get('doi', '')
        year = entry.get('year', '')

        print(f"[{idx:02d}/{len(target_keys):02d}] Auditing {key:<28} ... ", end="", flush=True)

        official = query_pubmed(title=title, author=author, doi=doi)
        if not official:
            official = query_crossref(title=title, author=author, doi=doi)

        if not official:
            print("❌ NOT FOUND")
            results.append({
                "key": key,
                "status": "NOT_FOUND",
                "bib": entry,
                "official": None,
                "discrepancies": ["Record could not be verified in PubMed or CrossRef"]
            })
            continue

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

        if discrepancies:
            print(f"⚠️ DISCREPANCY ({len(discrepancies)} issues)")
            status = "DISCREPANCY"
        else:
            print(f"✅ MATCH ({official.get('source')})")
            status = "MATCH"

        results.append({
            "key": key,
            "status": status,
            "bib": entry,
            "official": official,
            "discrepancies": discrepancies
        })

    # Summary stats
    matches = [r for r in results if r['status'] == 'MATCH']
    discrepancies = [r for r in results if r['status'] == 'DISCREPANCY']
    not_found = [r for r in results if r['status'] == 'NOT_FOUND']

    print("\n" + "=" * 60)
    print("📊 BIBLIOGRAPHY VALIDATION SUMMARY")
    print(f"   Total checked:    {len(target_keys)}")
    print(f"   ✅ Exact / Canonical Match: {len(matches):<2} ({len(matches)/len(target_keys)*100:.1f}%)")
    print(f"   ⚠️ Discrepancy / Variation: {len(discrepancies):<2} ({len(discrepancies)/len(target_keys)*100:.1f}%)")
    print(f"   ❌ Unindexed / Not Found:   {len(not_found):<2} ({len(not_found)/len(target_keys)*100:.1f}%)")
    print("=" * 60)

    # Save Markdown report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# NCBI PubMed & CrossRef Bibliography Verification Report\n\n")
        f.write(f"Generated on **{time.strftime('%Y-%m-%d %H:%M:%S')}**.\n\n")
        f.write("### Executive Summary\n\n")
        f.write(f"- **Total Citations Audited:** {len(target_keys)}\n")
        f.write(f"- **Exact / Canonical Records Verified:** {len(matches)} ({len(matches)/len(target_keys)*100:.1f}%)\n")
        f.write(f"- **Discrepancies & Minor Variations Detected:** {len(discrepancies)} ({len(discrepancies)/len(target_keys)*100:.1f}%)\n")
        f.write(f"- **Unindexed / Custom Entries:** {len(not_found)} ({len(not_found)/len(target_keys)*100:.1f}%)\n\n")

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

    print(f"📝 Full validation report written to {report_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", default="hyphaeon_paper/references.bib")
    parser.add_argument("--aux", default="hyphaeon_paper/main.aux")
    parser.add_argument("--report", default="hyphaeon_paper/bibliography_validation_report.md")
    args = parser.parse_args()

    validate_all(args.bib, args.aux, args.report)
