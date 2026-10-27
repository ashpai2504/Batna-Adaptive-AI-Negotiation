"""Download the IMF PortWatch tables used for real shocks -> data/real/

  portwatch_events.csv       GDACS-derived disruption events (type, dates, centroid, affected ports)
  portwatch_event_ports.csv  event -> affected port links with port coordinates and distance
  portwatch_ports.csv        port reference table (2,065 ports)
  portwatch_daily_calls.csv  daily port calls for every affected port, [from - 60d, to + 60d]
  placebo_daily_calls.csv    same ports, matching windows one year earlier (no-event control)

Source: IMF PortWatch public ArcGIS feature services (no key required),
https://portwatch.imf.org/ . Re-running overwrites the files.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real")
PAD_DAYS = 60
FIRST_DAY = pd.Timestamp("2019-01-01")


def query(service, params, retries=4):
    url = f"{BASE}/{service}/FeatureServer/0/query?" + urllib.parse.urlencode({"f": "json", **params})
    for k in range(retries):
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=90).read())
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch_all(service, where="1=1", fields="*", page=1000):
    rows, offset = [], 0
    while True:
        d = query(service, {"where": where, "outFields": fields, "resultOffset": offset,
                            "resultRecordCount": page, "orderByFields": "ObjectId"})
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < page and not d.get("exceededTransferLimit"):
            return pd.DataFrame(rows)
        offset += len(feats)


def ms_to_date(s):
    return pd.to_datetime(pd.to_numeric(s), unit="ms").dt.normalize()


def daily_calls(portid, start, end):
    where = f"portid='{portid}' AND date >= DATE '{start:%Y-%m-%d}' AND date <= DATE '{end:%Y-%m-%d}'"
    df = fetch_all("Daily_Ports_Data", where, "date,portid,portcalls,portcalls_container,portcalls_cargo,import,export")
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    events = fetch_all("portwatch_disruptions_database")
    events["fromdate"] = ms_to_date(events["fromdate"])
    events["todate"] = ms_to_date(events["todate"])
    events = events.drop(columns=[c for c in ("htmldescription", "Shape__Area", "Shape__Length", "pageid") if c in events])
    events.to_csv(os.path.join(OUT, "portwatch_events.csv"), index=False)

    links = fetch_all("disruptions_with_ports")
    links["fromdate"] = ms_to_date(links["fromdate"])
    links["todate"] = ms_to_date(links["todate"])
    links.to_csv(os.path.join(OUT, "portwatch_event_ports.csv"), index=False)

    ports = fetch_all("PortWatch_ports_database",
                      fields="portid,portname,country,ISO3,continent,lat,lon,vessel_count_total,vessel_count_container,LOCODE")
    ports.to_csv(os.path.join(OUT, "portwatch_ports.csv"), index=False)
    print(f"events={len(events)} links={len(links)} ports={len(ports)}")

    jobs = []
    for r in links.itertuples():
        start = max(FIRST_DAY, r.fromdate - pd.Timedelta(days=PAD_DAYS))
        end = r.todate + pd.Timedelta(days=PAD_DAYS)
        if end <= FIRST_DAY:
            continue
        jobs.append((r.eventid, r.portid, start, end))

    def run(job, shift_days=0):
        ev, port, s, e = job
        s2, e2 = s - pd.Timedelta(days=shift_days), e - pd.Timedelta(days=shift_days)
        if e2 <= FIRST_DAY:
            return None
        d = daily_calls(port, max(FIRST_DAY, s2), e2)
        if d.empty:
            return None
        d["eventid"] = ev
        return d

    with ThreadPoolExecutor(8) as ex:
        res = [d for d in ex.map(run, jobs) if d is not None]
    calls = pd.concat(res, ignore_index=True)
    calls.to_csv(os.path.join(OUT, "portwatch_daily_calls.csv"), index=False)
    print(f"daily rows={len(calls)} for {calls.groupby(['eventid', 'portid']).ngroups} event-port pairs")

    with ThreadPoolExecutor(8) as ex:
        res = [d for d in ex.map(lambda j: run(j, 364), jobs) if d is not None]
    placebo = pd.concat(res, ignore_index=True)
    placebo.to_csv(os.path.join(OUT, "placebo_daily_calls.csv"), index=False)
    print(f"placebo rows={len(placebo)}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
