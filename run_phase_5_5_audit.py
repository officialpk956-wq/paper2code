import sqlite3
import json
import time
import requests
import traceback
import os
import sys

BASE_URL = "http://127.0.0.1:8000"
ISSUES = []
PERFORMANCE = {}

# Make sure server is running
# try:
#     requests.get(f"{BASE_URL}/")
# except Exception as e:
#     print("Server not running. Please start the server.")
#     sys.exit(1)

print("Starting Phase 5.5 Audit...")

# AREA 7: API AUDIT & AREA 3: PERFORMANCE
def check_api():
    print("Running API checks...")
    
    # GET /api/papers
    t0 = time.time()
    try:
        res = requests.get(f"{BASE_URL}/api/papers")
        if res.status_code == 200:
            PERFORMANCE["Library load time"] = time.time() - t0
            papers = res.json()
            if len(papers) > 0:
                paper_id = papers[0]['id']
                
                # GET /api/papers/{id}
                t0 = time.time()
                res_p = requests.get(f"{BASE_URL}/api/papers/{paper_id}")
                if res_p.status_code == 200:
                    PERFORMANCE["Paper load time"] = time.time() - t0
                else:
                    ISSUES.append({"area": "API", "severity": "High", "desc": f"/api/papers/{paper_id} returned {res_p.status_code}"})
                
                # GET /api/papers/{id}/modules
                t0 = time.time()
                res_m = requests.get(f"{BASE_URL}/api/papers/{paper_id}/modules")
                if res_m.status_code == 200:
                    modules = res_m.json().get("modules", [])
                    if len(modules) > 0:
                        mod_id = modules[0]['id']
                        
                        # GET /api/modules/{id}
                        t0 = time.time()
                        res_mod = requests.get(f"{BASE_URL}/api/modules/{mod_id}")
                        if res_mod.status_code == 200:
                            PERFORMANCE["Module load time"] = time.time() - t0
                        else:
                            ISSUES.append({"area": "API", "severity": "High", "desc": f"/api/modules/{mod_id} returned {res_mod.status_code}"})
        else:
            ISSUES.append({"area": "API", "severity": "High", "desc": f"/api/papers returned {res.status_code}"})
    except Exception as e:
        ISSUES.append({"area": "API", "severity": "Critical", "desc": f"API request failed: {e}"})

    # POST /api/playground/generate
    t0 = time.time()
    try:
        payload = {"architecture": "ResNet", "config": {}}
        res = requests.post(f"{BASE_URL}/api/playground/generate", json=payload)
        if res.status_code == 200:
            PERFORMANCE["Playground graph regeneration time"] = time.time() - t0
        else:
            ISSUES.append({"area": "API", "severity": "Medium", "desc": f"Playground generate failed: {res.status_code}"})
    except Exception as e:
        pass # maybe not running

check_api()

# AREA 4: DATA QUALITY AUDIT
def check_db():
    print("Running DB checks...")
    try:
        conn = sqlite3.connect("c:\\papper2code\\tensortonic_dev.db")
        conn.row_factory = sqlite3.Row
        
        # Check Papers
        papers = conn.execute("SELECT * FROM papers").fetchall()
        for p in papers:
            if not p["title"]:
                ISSUES.append({"area": "Data", "severity": "High", "desc": f"Paper {p['id']} has empty title"})
                
        # Check Modules
        modules = conn.execute("SELECT * FROM paper_modules").fetchall()
        
        # duplicates
        seen_modules = set()
        for m in modules:
            # NaN check (if any float column has NaN)
            try:
                if "NaN" in str(m["flops_context"]):
                    ISSUES.append({"area": "Data", "severity": "Medium", "desc": f"NaN values in flops_context for module {m['id']}"})
            except:
                pass
                
            if not m["layer_name"]:
                ISSUES.append({"area": "Data", "severity": "Medium", "desc": f"Empty module name for module {m['id']}"})
            
            dup_key = f"{m['paper_id']}-{m['layer_name']}-{m['order_index']}"
            if dup_key in seen_modules:
                ISSUES.append({"area": "Data", "severity": "Low", "desc": f"Duplicate module detected: {dup_key}"})
            seen_modules.add(dup_key)
            
            if not m["explanation"] or len(m["explanation"].strip()) < 10:
                ISSUES.append({"area": "Data", "severity": "Low", "desc": f"Empty or very short explanation for module {m['id']} '{m['layer_name']}'"})
                
            # JSON checks
            try:
                g = json.loads(m["graph_nodes"]) if isinstance(m["graph_nodes"], str) else m["graph_nodes"]
            except:
                ISSUES.append({"area": "Data", "severity": "High", "desc": f"Invalid graph nodes JSON for module {m['id']}"})
                
        conn.close()
    except Exception as e:
        print(f"DB check failed: {e}")

check_db()

# Generate report
print("\n--- PERFORMANCE ---")
for k, v in PERFORMANCE.items():
    print(f"{k}: {v:.3f}s")
    
print("\n--- ISSUES ---")
if not ISSUES:
    print("None!")
for i in ISSUES:
    print(f"[{i['severity']}] {i['area']}: {i['desc']}")

with open("audit_results.json", "w") as f:
    json.dump({"issues": ISSUES, "performance": PERFORMANCE}, f, indent=2)

print("Audit complete.")
