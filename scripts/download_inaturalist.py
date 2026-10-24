"""Download a balanced, credited set of research-grade iNaturalist tick photos.

Three resumable steps, run in order (or all at once with ``all``):

    list    every research-grade observation of each species whose photos carry
            a Creative Commons licence -> observations.jsonl
    select  a balanced subset: per-species caps, at most PER_OBSERVER per
            photographer, every juvenile/larva kept -> manifest.csv
    photos  one 1024 px ("large") photo per selected observation, verified ->
            images/<photo_id>.jpg, plus CREDITS.csv

Splits are assigned by photographer (hash of user id), so one person's photos
never appear in both training and evaluation. Requests stay within
iNaturalist's published limits: about one API request per second and well
under 5 GB of photos per hour. Labels are community identifications, not
expert review.

    .venv/bin/python scripts/download_inaturalist.py all
"""

import argparse
import csv
import hashlib
import io
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from common import WORK

OUT = WORK / "dataset/inaturalist"
API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "TickPredict-science-fair/1.0 (github.com/EricLBuehler)"
LICENSES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd")
SEED = 20261019
PER_OBSERVER = 30
SPLITS = (("train", 80), ("dev", 10), ("test", 10))

# (iNaturalist taxon id, scientific name, common name, cap or None for all)
SPECIES = (
    (60598, "Ixodes scapularis", "blacklegged tick", 3000),
    (60599, "Ixodes pacificus", "western blacklegged tick", None),
    (52155, "Dermacentor variabilis", "American dog tick", 3000),
    (83744, "Amblyomma americanum", "lone star tick", 3000),
    (180017, "Amblyomma maculatum", "Gulf Coast tick", 3000),
    (229905, "Rhipicephalus sanguineus", "brown dog tick", None),
    (297285, "Dermacentor andersoni", "Rocky Mountain wood tick", None),
    (485044, "Haemaphysalis longicornis", "Asian longhorned tick", None),
)

# Controlled annotation ids (GET /v1/controlled_terms).
LIFE_STAGE, ADULT, LARVA, JUVENILE = 1, 2, 6, 8
SEX, FEMALE, MALE = 9, 10, 11


def _get(url, attempts=6):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            if status in (403, 404) or attempt == attempts - 1:
                raise
            time.sleep(min(120, 5 * 2 ** attempt) if status == 429 else 2 * (attempt + 1))


# ---------------------------------------------------------------- list

def _slim(observation):
    """Keep only what training and attribution need (no coordinates)."""
    return {
        "id": observation["id"],
        "uri": observation["uri"],
        "observed_on": observation.get("observed_on"),
        "taxon_id": observation["taxon"]["id"],
        "taxon_name": observation["taxon"]["name"],
        "user_id": observation["user"]["id"],
        "user_login": observation["user"]["login"],
        "agreements": observation.get("num_identification_agreements"),
        "disagreements": observation.get("num_identification_disagreements"),
        "annotations": [[a["controlled_attribute_id"], a["controlled_value_id"], a.get("vote_score", 0)]
                        for a in observation.get("annotations", [])],
        "photos": [{"id": p["id"], "url": p["url"], "license": p.get("license_code"),
                    "attribution": p.get("attribution"), "size": p.get("original_dimensions")}
                   for p in observation.get("photos", [])],
    }


def list_species(taxon_id, name):
    path = OUT / "observations" / f"{taxon_id}.jsonl"
    done = path.with_suffix(".done")
    if done.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_id = 0
    if path.exists():
        with path.open() as file:
            for line in file:
                last_id = max(last_id, json.loads(line)["id"])
    total = None
    with path.open("a") as file:
        while True:
            query = urllib.parse.urlencode({
                "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
                "photo_license": ",".join(LICENSES), "order_by": "id", "order": "asc",
                "id_above": last_id, "per_page": 200})
            page = json.loads(_get(f"{API}?{query}"))
            total = total or page["total_results"]
            results = page["results"]
            for observation in results:
                file.write(json.dumps(_slim(observation)) + "\n")
            file.flush()
            if not results:
                break
            last_id = results[-1]["id"]
            print(f"  {name}: through id {last_id} ({total} total)", flush=True)
            time.sleep(1.1)
    done.write_text("complete\n")


# ---------------------------------------------------------------- select

def _labels(annotations):
    """Life stage and sex from community annotations with a non-negative vote."""
    values = {(term, value) for term, value, score in annotations if score >= 0}
    if (LIFE_STAGE, LARVA) in values:
        stage = "larva"
    elif (LIFE_STAGE, JUVENILE) in values:
        stage = "nymph"
    elif (LIFE_STAGE, ADULT) in values:
        stage = "adult"
    else:
        stage = ""
    sex = "female" if (SEX, FEMALE) in values else "male" if (SEX, MALE) in values else ""
    if stage == "adult" and sex:
        stage = f"adult {sex}"
    return stage, sex


def _split(user_id):
    bucket = int(hashlib.sha256(f"{SEED}:{user_id}".encode()).hexdigest(), 16) % 100
    for name, share in SPLITS:
        if bucket < share:
            return name
        bucket -= share
    raise AssertionError


def select():
    rows = []
    for taxon_id, name, common, cap in SPECIES:
        candidates = []
        with (OUT / "observations" / f"{taxon_id}.jsonl").open() as file:
            for line in file:
                observation = json.loads(line)
                photo = next((p for p in observation["photos"] if p["license"] in LICENSES), None)
                if photo is None:
                    continue
                stage, sex = _labels(observation["annotations"])
                candidates.append((observation, photo, stage, sex))
        rng = random.Random(f"{SEED}:{taxon_id}")
        rng.shuffle(candidates)
        # Scarce stages first, then any annotated record, then the rest.
        candidates.sort(key=lambda c: 0 if c[2] in ("nymph", "larva") else 1 if c[2] or c[3] else 2)
        per_user = Counter()
        chosen = []
        for observation, photo, stage, sex in candidates:
            if per_user[observation["user_id"]] >= PER_OBSERVER:
                continue
            if cap is not None and len(chosen) >= cap and stage not in ("nymph", "larva"):
                continue
            per_user[observation["user_id"]] += 1
            chosen.append((observation, photo, stage, sex))
        for observation, photo, stage, sex in chosen:
            rows.append({
                "observation_id": observation["id"], "photo_id": photo["id"],
                "species": name, "common_name": common, "taxon_name": observation["taxon_name"],
                "life_stage": stage, "sex": sex, "observed_on": observation["observed_on"] or "",
                "user_id": observation["user_id"], "split": _split(observation["user_id"]),
                "license": photo["license"], "attribution": photo["attribution"] or "",
                "observation_url": observation["uri"],
                "photo_url": photo["url"].replace("/square.", "/large."),
            })
        print(f"{name}: {len(chosen)} of {len(candidates)} licensed observations, "
              f"{len(per_user)} photographers", flush=True)
    with (OUT / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {len(rows)} observations; splits {dict(Counter(r['split'] for r in rows))}")


# ---------------------------------------------------------------- photos

class RateLimit:
    """At most `per_second` starts per second and `bytes_per_hour` downloaded."""

    def __init__(self, per_second, bytes_per_hour):
        self.interval = 1 / per_second
        self.bytes_per_hour = bytes_per_hour
        self.lock = threading.Lock()
        self.next_start = time.monotonic()
        self.started = time.monotonic()
        self.bytes = 0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            budget_time = self.started + self.bytes / self.bytes_per_hour * 3600
            start = max(self.next_start, budget_time, now)
            self.next_start = start + self.interval
        time.sleep(max(0, start - time.monotonic()))

    def add(self, count):
        with self.lock:
            self.bytes += count


def photos(workers=4, per_second=2.0, gb_per_hour=3.0):
    with (OUT / "manifest.csv").open() as file:
        rows = list(csv.DictReader(file))
    images = OUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    limit = RateLimit(per_second, gb_per_hour * 1e9)
    failures = []
    counter = Counter()
    lock = threading.Lock()

    def fetch(row):
        target = images / f"{row['photo_id']}.jpg"
        if target.exists():
            with lock:
                counter["present"] += 1
            return
        limit.wait()
        try:
            data = _get(row["photo_url"])
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            temporary = target.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.rename(target)
            limit.add(len(data))
            with lock:
                counter["downloaded"] += 1
                done = counter["downloaded"] + counter["present"]
                if done % 250 == 0:
                    print(f"  {done}/{len(rows)} photos, {limit.bytes / 1e9:.2f} GB this run", flush=True)
        except Exception as error:  # recorded and retried on the next run
            with lock:
                failures.append({"photo_id": row["photo_id"], "error": repr(error)})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, rows))
    (OUT / "photo_failures.json").write_text(json.dumps(failures, indent=1) + "\n")
    with (OUT / "CREDITS.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["photo_id", "species", "attribution", "license", "observation_url"])
        for row in rows:
            writer.writerow([row["photo_id"], row["species"], row["attribution"], row["license"],
                             row["observation_url"]])
    print(f"Photos: {dict(counter)}, failures {len(failures)}; credits in {OUT / 'CREDITS.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("list", "select", "photos", "all"))
    args = parser.parse_args()
    if args.step in ("list", "all"):
        for taxon_id, name, _, _ in SPECIES:
            print(f"Listing {name}", flush=True)
            list_species(taxon_id, name)
    if args.step in ("select", "all"):
        select()
    if args.step in ("photos", "all"):
        photos()


if __name__ == "__main__":
    main()
