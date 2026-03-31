#!/usr/bin/env python3
"""
Populate addresses.csv with residential homes near a reference address.

Steps:
  1. Geocode the reference address via Google Maps API
  2. Query OpenStreetMap Overpass API for all residential addresses within --radius meters
  3. Write results to addresses.csv (address, lat, lng)

Usage:
    python populate_addresses.py --reference "9212 Cedarcrest Dr, Dallas, TX" --limit 100
    python populate_addresses.py --reference "9212 Cedarcrest Dr" --limit 50 --radius 300
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

# ── Load API key from .env ─────────────────────────────────────────────────────

def load_env(path: str = ".env") -> dict:
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env

_env = load_env()
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY") or _env.get("GOOGLE_MAPS_API_KEY")

GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"
OVERPASS_URLS  = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# ──────────────────────────────────────────────────────────────────────────────

def geocode_reference(address: str) -> tuple[float, float, str]:
    """Return (lat, lng, formatted_address) for the reference address."""
    params = urllib.parse.urlencode({"address": address, "key": API_KEY})
    with urllib.request.urlopen(f"{GEOCODE_URL}?{params}") as resp:
        data = json.loads(resp.read())
    if data["status"] != "OK":
        sys.exit(f"Geocoding failed for '{address}': {data['status']}")
    result = data["results"][0]
    loc    = result["geometry"]["location"]
    return loc["lat"], loc["lng"], result["formatted_address"]


def query_overpass(lat: float, lng: float, radius: int) -> list[dict]:
    """
    Query Overpass API for all nodes/ways with address tags within `radius` meters.
    Returns list of dicts: {address, lat, lng}
    """
    query = (
        f"[out:json][timeout:30];"
        f"("
        f'node["addr:housenumber"]["addr:street"](around:{radius},{lat},{lng});'
        f'way["addr:housenumber"]["addr:street"](around:{radius},{lat},{lng});'
        f'relation["addr:housenumber"]["addr:street"](around:{radius},{lat},{lng});'
        f");"
        f"out center;"
    )
    # Overpass expects: POST body = data=<url-encoded-query>
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")

    last_error = None
    for url in OVERPASS_URLS:
        try:
            request = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(request, timeout=60) as resp:
                result = json.loads(resp.read())
            break
        except Exception as e:
            print(f"  [warn] {url} failed: {e} — trying next mirror …")
            last_error = e
            time.sleep(1)
    else:
        raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_error}")

    homes = []
    seen  = set()

    for element in result.get("elements", []):
        tags = element.get("tags", {})
        house_number = tags.get("addr:housenumber", "").strip()
        street       = tags.get("addr:street", "").strip()
        city         = tags.get("addr:city", "").strip()
        state        = tags.get("addr:state", "").strip()
        postcode     = tags.get("addr:postcode", "").strip()

        if not house_number or not street:
            continue

        # Build a human-readable address
        parts   = [f"{house_number} {street}"]
        if city:     parts.append(city)
        if state:    parts.append(state)
        if postcode: parts.append(postcode)
        address = ", ".join(parts)

        # Deduplicate
        key = (house_number.lower(), street.lower())
        if key in seen:
            continue
        seen.add(key)

        # Coordinates: nodes have lat/lng directly; ways/relations expose center
        if element["type"] == "node":
            elat, elng = element["lat"], element["lon"]
        else:
            center = element.get("center", {})
            elat   = center.get("lat", lat)
            elng   = center.get("lon", lng)

        homes.append({"address": address, "lat": elat, "lng": elng})

    return homes


def write_csv(rows: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["address", "lat", "lng"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not API_KEY:
        sys.exit("GOOGLE_MAPS_API_KEY not found in .env or environment.")

    parser = argparse.ArgumentParser(
        description="Populate addresses.csv with homes near a reference address."
    )
    parser.add_argument(
        "--reference", required=True,
        help='Reference address, e.g. "9212 Cedarcrest Dr, Dallas, TX"'
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max number of homes to include (default: 100)"
    )
    parser.add_argument(
        "--radius", type=int, default=400,
        help="Search radius in meters around reference address (default: 400)"
    )
    parser.add_argument(
        "--out", default="addresses.csv",
        help="Output CSV file (default: addresses.csv)"
    )
    args = parser.parse_args()

    # 1. Geocode reference address
    print(f"Geocoding reference: {args.reference} …")
    ref_lat, ref_lng, formatted = geocode_reference(args.reference)
    print(f"  → {formatted}  ({ref_lat:.6f}, {ref_lng:.6f})")

    # 2. Query Overpass for nearby homes
    print(f"\nQuerying OpenStreetMap for addresses within {args.radius}m …")
    homes = query_overpass(ref_lat, ref_lng, args.radius)
    print(f"  → Found {len(homes)} address(es)")

    if not homes:
        print("\nNo addresses found. Try increasing --radius.")
        sys.exit(0)

    # 3. Apply limit
    if len(homes) > args.limit:
        print(f"  → Trimming to {args.limit} (--limit)")
        homes = homes[: args.limit]

    # 4. Write CSV
    write_csv(homes, args.out)
    print(f"\nWrote {len(homes)} addresses to {args.out}")
    print("Run fetch_aerial_views.py next to download satellite images.")


if __name__ == "__main__":
    main()
