#!/usr/bin/env python3
"""
show_reid_data.py
=================
CLI utility script to display all registered Re-ID identities, track IDs, 
and cross-camera histories stored in the persisted database.
Use this script to verify and prove that Re-ID is working correctly to your client.
"""

import json
from pathlib import Path

def print_reid_summary():
    id_map_path = Path("data/embeddings/identity_map.json")
    faiss_path = Path("data/embeddings/faiss.index")

    print("=" * 80)
    print("                URG-IS GLOBAL RE-IDENTIFICATION SYSTEM PROOF DATABASE")
    print("=" * 80)

    # 1. Check FAISS Index
    if faiss_path.exists():
        print(f"[STATUS] FAISS Index File Detected: {faiss_path} (Size: {faiss_path.stat().st_size} bytes)")
    else:
        print(f"[STATUS] FAISS Index File: Not Found (or pipeline has not run yet)")

    # 2. Check Identity Map
    if not id_map_path.exists():
        print("\n[WARNING] Identity Map file not found at: data/embeddings/identity_map.json")
        print("To generate this database:")
        print("  1. Start the Streamlit dashboard: streamlit run mvp_dashboard.py")
        print("  2. Start the tracking pipeline on camera streams from the Home tab.")
        print("  3. Allow it to run for a few seconds so that identities are registered.")
        print("  4. Stop the pipeline (or stop the app) to persist the tracking data to disk.")
        print("=" * 80)
        return

    try:
        with open(id_map_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n[ERROR] Failed to read identity map JSON: {e}")
        print("=" * 80)
        return

    identities = data.get("identities", {})
    next_id = data.get("next_id", 1)
    vector_to_person = data.get("vector_to_person", [])

    print(f"[STATUS] Database Mapping File Loaded: {id_map_path}")
    print(f"[STATS] Total Unique Identities Registered : {len(identities)}")
    print(f"[STATS] Total FAISS Feature Vectors Stored  : {len(vector_to_person)}")
    print(f"[STATS] Next Available Auto-Identity Index  : {next_id}")
    print("-" * 80)

    if not identities:
        print("\nNo unique identities have been registered in the mapping database yet.")
        print("=" * 80)
        return

    # Print Table Header
    header = f"{'PERSON ID':<12} | {'TRACKER IDS SEEN':<22} | {'CAMERA PATHWAY':<18} | {'VECTORS':<8} | {'FIRST FRAME':<12}"
    print(header)
    print("-" * 80)

    for person_id, details in sorted(identities.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        pid_label = f"P{person_id}"
        
        # Format trackers and cameras
        trackers = ", ".join(map(str, details.get("track_ids", [])))
        cameras = " -> ".join(sorted(details.get("camera_ids", [])))
        
        vectors = details.get("embedding_count", 0)
        first_seen = details.get("first_seen", 0)
        
        # Truncate lists if too long
        if len(trackers) > 20:
            trackers = trackers[:17] + "..."
            
        print(f"{pid_label:<12} | {trackers:<22} | {cameras:<18} | {vectors:<8} | {first_seen:<12}")

    print("=" * 80)
    print("NOTE: Appearance fingerprints (512-dim vectors) are matched dynamically in FAISS.")
    print("When the same person appears on a different camera, FAISS maps their tracker ID")
    print("to the existing stable PERSON ID, demonstrating cross-camera Re-ID in real time.")
    print("=" * 80)

if __name__ == "__main__":
    print_reid_summary()
