"""
validate.py — Pre-submission validation script
Run this before submitting to catch any issues.
Usage: python validate.py --space-url http://localhost:7860
"""

import sys
import json
import argparse
import requests

def check(label, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status} — {label}")
    if detail and not condition:
        print(f"         → {detail}")
    return condition


def validate(space_url: str):
    base = space_url.rstrip("/")
    results = []
    print(f"\n🏥 ICU Alarm Fatigue Reducer — Pre-Submission Validation")
    print(f"   Target: {base}\n")

    # 1. Health check
    try:
        r = requests.get(f"{base}/health", timeout=10)
        results.append(check("Space responds to /health", r.status_code == 200))
    except Exception as e:
        results.append(check("Space responds to /health", False, str(e)))
        print("  ⚠ Cannot reach space. Make sure it's running.\n")
        return False

    # 2–4. Test each task level
    for level in ["easy", "medium", "hard"]:
        try:
            # Reset
            r = requests.post(f"{base}/reset", params={"task_level": level, "session_id": f"val_{level}"}, timeout=15)
            obs_ok = r.status_code == 200
            obs = r.json() if obs_ok else {}
            results.append(check(f"/reset returns observation for '{level}'", obs_ok and "prompt" in obs))

            # Step
            action = {"classification": "false", "confidence": 0.6,
                      "explanation": "Test validation run", "recommended_action": "Continue monitoring"}
            r2 = requests.post(f"{base}/step", json=action,
                               params={"session_id": f"val_{level}"}, timeout=15)
            step_ok = r2.status_code == 200
            step_data = r2.json() if step_ok else {}
            score = step_data.get("reward", -1)
            results.append(check(f"/step returns reward for '{level}'", step_ok and "reward" in step_data))
            results.append(check(f"Score in range [0.0-1.0] for '{level}'", 0.0 <= score <= 1.0,
                                 f"Got score: {score}"))

            # State
            r3 = requests.get(f"{base}/state", params={"session_id": f"val_{level}"}, timeout=10)
            results.append(check(f"/state works for '{level}'", r3.status_code == 200))

        except Exception as e:
            results.append(check(f"Task '{level}' endpoints", False, str(e)))

    # 5. Verify openenv.yaml exists
    import os
    results.append(check("openenv.yaml exists", os.path.exists("openenv.yaml")))
    results.append(check("inference.py exists", os.path.exists("inference.py")))
    results.append(check("Dockerfile exists", os.path.exists("Dockerfile")))
    results.append(check("README.md exists", os.path.exists("README.md")))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*45}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉 ALL CHECKS PASSED — Ready to submit!")
    else:
        print(f"  ⚠ {total - passed} check(s) failed — fix before submitting")
    print(f"{'='*45}\n")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-url", default="http://localhost:7860",
                        help="URL of your HuggingFace space or local server")
    args = parser.parse_args()
    success = validate(args.space_url)
    sys.exit(0 if success else 1)
