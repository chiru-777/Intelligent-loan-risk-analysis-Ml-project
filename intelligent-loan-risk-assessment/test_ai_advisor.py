"""
Test script to verify AI Credit Advisor diagnostic & matter generation.
"""
import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.ai_advisor import AICreditAdvisor
from predict import get_sample_applicant

def test_advisor():
    advisor = AICreditAdvisor()

    # 1. High Risk Profile Test
    print("=" * 70)
    print("  TESTING AI ADVISOR: HIGH-RISK APPLICANT PROFILE")
    print("=" * 70)
    high_app = get_sample_applicant('high')
    res_high = {'prediction': 'High Risk', 'risk_percentage': 98.81, 'prob_high_risk': 0.9881}
    report_high = advisor.generate_full_diagnostic_report(high_app, res_high)

    print(f"Risk Tier  : {report_high['risk_tier']}")
    print(f"Action Code: {report_high['action_code']}")
    print(f"\nProblems Detected ({len(report_high['problems'])}):")
    for p in report_high['problems']:
        print(f"  • [{p['severity'].upper()}] {p['category']}: Observed {p['metric']} (Target: {p['benchmark']})")
        print(f"    Variance: {p['deficit']}")
        print(f"    Diagnosis: {p['explanation']}\n")

    print("Restructuring Action Plan:")
    for step in report_high['action_plan']:
        print(f"  {step}")

    # 2. Low Risk Profile Test
    print("\n" + "=" * 70)
    print("  TESTING AI ADVISOR: LOW-RISK APPLICANT PROFILE")
    print("=" * 70)
    low_app = get_sample_applicant('low')
    res_low = {'prediction': 'Low Risk', 'risk_percentage': 1.29, 'prob_high_risk': 0.0129}
    report_low = advisor.generate_full_diagnostic_report(low_app, res_low)

    print(f"Risk Tier  : {report_low['risk_tier']}")
    print(f"Action Code: {report_low['action_code']}")
    print(f"\nStrengths Identified ({len(report_low['strengths'])}):")
    for s in report_low['strengths']:
        print(f"  • {s['category']} ({s['metric']}): {s['impact']}")

    print("\n[+] AI Credit Advisor test passed successfully!")

if __name__ == '__main__':
    test_advisor()
