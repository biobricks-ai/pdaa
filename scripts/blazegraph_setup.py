"""
Minimal Blazegraph upload script
Requires:
  • simgraph.nt         # any RDF file you want to load (N-Triples, Turtle, RDF/XML …)
  • a Blazegraph server listening on http://localhost:9999
       e.g.  docker run -d -p 9999:9999 ghcr.io/blazegraph/database:2.1.5
"""
import requests
from pathlib import Path

# ---- settings ------------------------------------------------------------
NS_NAME        = "pdaa"                                # namespace name
BLAZE_BASE     = "http://localhost:9999"               # adjust host/port if different
NS_ROOT        = f"{BLAZE_BASE}/bigdata/namespace"     # /bigdata or /blazegraph depending on build
RDF_PATH       = Path("cache/associate_properties_with_aopwiki/simgraph.nt")  # your RDF file on disk
# -------------------------------------------------------------------------

# 1) (optional) delete the namespace if it already exists
requests.delete(f"{NS_ROOT}/{NS_NAME}", timeout=30)

# 2) create the namespace with minimal properties
NS_PROPERTIES = f"""<?xml version="1.0"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
<properties>
  <entry key="com.bigdata.rdf.sail.namespace">{NS_NAME}</entry>
</properties>"""
requests.post(NS_ROOT,
              headers={"Content-Type": "application/xml"},
              data=NS_PROPERTIES,
              timeout=30).raise_for_status()

# 3) upload the RDF payload
with RDF_PATH.open("rb") as fp:
    requests.post(f"{NS_ROOT}/{NS_NAME}/sparql",
                  headers={"Content-Type": "application/x-turtle"},
                  data=fp,
                  timeout=120).raise_for_status()

print("✓ Blazegraph namespace created and data loaded")
