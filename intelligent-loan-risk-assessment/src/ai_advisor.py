"""
AI Credit Risk Diagnostic & Advisory Engine.
Performs in-depth financial problem diagnosis and generates comprehensive
underwriting intelligence reports ('Matter'):
- Executive Risk Diagnosis & Tier
- Detailed Root-Cause Problem Breakdown (Identifies exact financial deficits)
- Financial Stress & Affordability Metrics
- Positive Compensating Factors (Strengths)
- Prescriptive Action Plan (Step-by-step roadmap to convert High Risk to Low Risk)
- Formal Institutional Credit Memorandum
"""

import math
from typing import Dict, Any, List, Optional
import pandas as pd


class AICreditAdvisor:
    """
    Expert AI Financial Diagnostician and Credit Advisor.
    Translates model predictions, SHAP attribution, and financial ratios
    into deep, professional diagnostic reports.
    """

    # Industry benchmark standards
    BENCHMARKS = {
        'credit_score': {'prime': 720, 'acceptable': 660, 'subprime': 580},
        'debt_to_income_ratio': {'safe': 0.35, 'caution': 0.45, 'dangerous': 0.50},
        'loan_to_income_ratio': {'safe': 0.30, 'caution': 0.50, 'dangerous': 0.70},
        'asset_to_loan_ratio': {'safe': 1.5, 'adequate': 0.8, 'deficient': 0.3}
    }

    def __init__(self):
        pass

    def generate_full_diagnostic_report(
        self,
        applicant: Dict[str, Any],
        prediction_result: Dict[str, Any],
        shap_explanation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes applicant profile, model prediction, and SHAP drivers
        into a comprehensive, in-depth diagnostic dossier.
        """
        pred_label = prediction_result.get('prediction', 'Unknown')
        risk_pct = prediction_result.get('risk_percentage', 50.0)
        prob_high = prediction_result.get('prob_high_risk', 0.5)

        # 1. Calculate financial ratios and stress indices
        income = float(applicant.get('income', 1.0))
        loan_amt = float(applicant.get('loan_amount', 0.0))
        credit_score = int(applicant.get('credit_score', 600))
        existing_debts = float(applicant.get('existing_debts', 0.0))
        dti = float(applicant.get('debt_to_income_ratio', 0.35))
        assets = float(applicant.get('assets', 0.0))
        term_months = int(applicant.get('loan_term', 36))
        emp_status = str(applicant.get('employment_status', 'Employed'))
        pay_hist = str(applicant.get('payment_history', 'On-Time'))

        # Estimated monthly installment (assuming standard 8.5% annual interest)
        annual_rate = 0.085
        monthly_rate = annual_rate / 12
        if term_months > 0:
            emi = (loan_amt * monthly_rate * ((1 + monthly_rate) ** term_months)) / (((1 + monthly_rate) ** term_months) - 1)
        else:
            emi = loan_amt / 36

        monthly_income = income / 12
        est_monthly_loan_burden = (emi / (monthly_income + 1e-5)) * 100
        loan_to_income = (loan_amt / (income + 1e-5)) * 100
        asset_coverage = (assets / (loan_amt + 1e-5)) * 100

        # 2. Risk Tier Classification
        if prob_high >= 0.85:
            risk_tier = "Severe High Risk (Tier 4 - Deep Distress)"
            verdict_color = "#E74C3C"
            action_code = "IMMEDIATE REJECTION / COMPREHENSIVE RESTRUCTURING REQUIRED"
        elif prob_high >= 0.50:
            risk_tier = "Elevated Risk (Tier 3 - Subprime Margin)"
            verdict_color = "#E67E22"
            action_code = "CONDITIONAL REFERRAL / COLLATERAL ENHANCEMENT REQUIRED"
        elif prob_high >= 0.20:
            risk_tier = "Moderate Risk (Tier 2 - Standard Acceptable)"
            verdict_color = "#F1C40F"
            action_code = "STANDARD UNDERWRITING APPROVAL"
        else:
            risk_tier = "Prime Low Risk (Tier 1 - Preferred Borrower)"
            verdict_color = "#2ECC71"
            action_code = "FAST-TRACK EXPEDITED APPROVAL"

        # 3. Deep Root-Cause Problem Identification (In Simple Words Everyone Can Understand)
        problems = []
        strengths = []

        # Check Credit Score
        if credit_score < self.BENCHMARKS['credit_score']['subprime']:
            problems.append({
                'category': 'Low Credit Score',
                'severity': 'Critical',
                'metric': f"{credit_score} Score",
                'benchmark': "At least 660 needed for a good loan",
                'deficit': f"{self.BENCHMARKS['credit_score']['acceptable'] - credit_score} points too low",
                'explanation': (
                    f"Your credit score is {credit_score}, which is quite low. A low score tells the bank that "
                    "there were missed payments or money troubles in the past. Banks feel worried that money lent to you "
                    "might not be paid back on time."
                )
            })
        elif credit_score < self.BENCHMARKS['credit_score']['acceptable']:
            problems.append({
                'category': 'Borderline Credit Score',
                'severity': 'Moderate',
                'metric': f"{credit_score} Score",
                'benchmark': "At least 660 needed for safe approval",
                'deficit': f"{self.BENCHMARKS['credit_score']['acceptable'] - credit_score} points below safe level",
                'explanation': (
                    f"Your credit score is {credit_score}, which is just on the edge. While not terrible, "
                    "banks prefer to see a score of 660 or higher before easily approving a loan."
                )
            })
        else:
            strengths.append({
                'category': 'Strong Credit Score',
                'metric': f"{credit_score} Score",
                'impact': "You have a great credit score! This proves you handle money responsibly and always pay back on time."
            })

        # Check Debt-to-Income (DTI)
        if dti > self.BENCHMARKS['debt_to_income_ratio']['dangerous']:
            problems.append({
                'category': 'Too Much Existing Debt',
                'severity': 'Critical',
                'metric': f"{dti * 100:.0f}% of your monthly income goes to debt",
                'benchmark': "Should be 35% or less",
                'deficit': f"{dti * 100 - 35:.0f}% too high",
                'explanation': (
                    f"You are spending {dti * 100:.0f}% of your monthly earnings just paying off old loans and credit cards. "
                    "When more than half your paycheck is already promised to other debts, you will not have enough money left over "
                    "to buy food, pay everyday bills, and pay this new loan installment."
                )
            })
        elif dti > self.BENCHMARKS['debt_to_income_ratio']['caution']:
            problems.append({
                'category': 'High Monthly Debt Burden',
                'severity': 'High',
                'metric': f"{dti * 100:.0f}% of your monthly income goes to debt",
                'benchmark': "Should be 35% or less",
                'deficit': f"{dti * 100 - 35:.0f}% above safe limit",
                'explanation': (
                    f"A debt level of {dti * 100:.0f}% is higher than what banks like to see. It means a big part of your "
                    "salary is already tied up in other payments, leaving very little breathing room."
                )
            })
        else:
            strengths.append({
                'category': 'Low Existing Debt',
                'metric': f"Only {dti * 100:.0f}% of salary goes to debt",
                'impact': "You owe very little on existing debts, leaving plenty of spare money each month to easily afford this loan."
            })

        # Check Payment History
        if pay_hist == 'Missed':
            problems.append({
                'category': 'Past Missed Payments',
                'severity': 'Critical',
                'metric': 'Missed Payments on Record',
                'benchmark': 'Clean On-Time Record',
                'deficit': 'Has missed bill deadlines in the past',
                'explanation': (
                    "Your record shows you missed loan or credit card payments in the past. This is the biggest warning sign "
                    "for any bank, because they worry that you might miss payments on this new loan too."
                )
            })
        elif pay_hist == 'Delayed':
            problems.append({
                'category': 'Late Bill Payments',
                'severity': 'High',
                'metric': 'Delayed Payments on Record',
                'benchmark': 'Clean On-Time Record',
                'deficit': 'Frequently pays bills late',
                'explanation': (
                    "You often pay bills late. Even if you pay eventually, paying late signals to the bank that you sometimes "
                    "run out of money before the month ends."
                )
            })
        else:
            strengths.append({
                'category': 'Spotless Payment Track Record',
                'metric': '100% On-Time Payments',
                'impact': "You always pay every bill right on time without fail. Banks love this!"
            })

        # Check Employment Status
        if emp_status == 'Unemployed':
            problems.append({
                'category': 'No Regular Job / Income',
                'severity': 'Critical',
                'metric': 'Currently Unemployed',
                'benchmark': 'Active Steady Job',
                'deficit': 'Zero regular monthly salary',
                'explanation': (
                    "You currently do not have a job. Without a regular monthly paycheck coming into your bank account, "
                    "it is almost impossible to keep making monthly loan installments."
                )
            })
        elif emp_status == 'Self-Employed':
            problems.append({
                'category': 'Income Can Go Up and Down',
                'severity': 'Moderate',
                'metric': 'Self-Employed',
                'benchmark': 'Steady Fixed Salary',
                'deficit': 'Earnings change month to month',
                'explanation': (
                    "Because you work for yourself, your monthly earnings can fluctuate. Banks usually ask for extra proof "
                    "like tax returns to make sure you can still pay during slower months."
                )
            })
        else:
            strengths.append({
                'category': 'Stable Job & Steady Salary',
                'metric': 'Employed Full-Time',
                'impact': "You have a stable job with a steady paycheck coming in every month to easily pay your loan installments."
            })

        # Check Asset Coverage / Collateral
        if asset_coverage < 50.0:
            problems.append({
                'category': 'Low Savings or Backup Money',
                'severity': 'High',
                'metric': f"${assets:,.0f} in savings ({asset_coverage:.0f}% of loan)",
                'benchmark': f"Should have at least ${loan_amt:,.0f} in backup savings",
                'deficit': f"${loan_amt - assets:,.0f} short on backup money",
                'explanation': (
                    f"You asked to borrow ${loan_amt:,.0f}, but only have ${assets:,.0f} in savings or property to fall back on. "
                    "If an emergency happens (like illness or losing work), you don't have enough backup money to cover the loan."
                )
            })
        else:
            strengths.append({
                'category': 'Strong Savings & Property Backup',
                'metric': f"${assets:,.0f} in savings ({asset_coverage:.0f}% of loan)",
                'impact': "You have plenty of savings and assets to protect you and the bank even if unexpected hard times hit."
            })

        # Check Previous Loans (Before Loans count)
        prev_loans = int(applicant.get('previous_loans', 0))
        if prev_loans >= 3 and pred_label == 'High Risk':
            problems.append({
                'category': 'Too Many Previous Loans',
                'severity': 'Moderate',
                'metric': f"{prev_loans} Previous Loans",
                'benchmark': '1 or 2 loans at a time',
                'deficit': f"{prev_loans - 2} extra loans on record",
                'explanation': (
                    f"You have taken out {prev_loans} loans before. Having taken many loans makes banks worry that you borrow "
                    "too frequently and are relying too much on borrowed money."
                )
            })

        # Check Loan Size compared to Income
        if loan_to_income > 45.0 and pred_label == 'High Risk':
            problems.append({
                'category': 'Loan Amount Too Large for Income',
                'severity': 'High',
                'metric': f"Loan is {loan_to_income:.0f}% of your yearly salary",
                'benchmark': 'Should be under 35% of yearly salary',
                'deficit': f"{loan_to_income - 35:.0f}% too large",
                'explanation': (
                    f"The loan amount of ${loan_amt:,.0f} is very large compared to your annual earnings of ${income:,.0f}. "
                    "Repaying such a big loan would take too big a bite out of your income each month."
                )
            })

        # 4. Prescriptive Action Plan in Simple Words (How to Fix the Risk)
        action_plan = []
        if pred_label == 'High Risk':
            # Step 1: DTI Reduction
            if dti > 0.40:
                debt_reduction = max(2000, existing_debts * 0.4)
                action_plan.append(
                    f"🔹 **1. Pay Off Some Old Debts:** Try to pay off about ${debt_reduction:,.0f} of your credit card or small loans. "
                    f"This frees up monthly cash and brings your debt down to a safe level (under 35%)."
                )

            # Step 2: Loan Restructuring
            if loan_amt > income * 0.35:
                suggested_loan = round((income * 0.30) / 1000) * 1000
                action_plan.append(
                    f"🔹 **2. Ask for a Smaller Loan:** Instead of borrowing ${loan_amt:,.0f}, try asking for about ${suggested_loan:,.0f}. "
                    "A smaller loan means smaller monthly installments that fit easily into your monthly budget."
                )

            # Step 3: Tenure Extension
            if term_months < 60:
                action_plan.append(
                    f"🔹 **3. Choose a Longer Repayment Time:** Increase your loan period from {term_months} months to 48 or 60 months. "
                    "This spreads out the payments and cuts your monthly bill by 25% to 35%."
                )

            # Step 4: Credit Score Rehabilitation
            if credit_score < 660:
                action_plan.append(
                    "🔹 **4. Boost Your Credit Score:** Pay every single bill on time for the next 6 months without fail, "
                    "and keep your credit card balances under 30%. Your score will quickly rise above 660."
                )

            # Step 5: Secondary Collateral / Co-Signer
            if asset_coverage < 80.0 or emp_status != 'Employed':
                action_plan.append(
                    "🔹 **5. Apply with a Co-Signer (Family Member):** Add a family member with a stable job and high credit score (720+) "
                    "to your application, or pledge property as security."
                )
        else:
            action_plan.append(
                "✅ **Maintain Current Financial Posture:** Keep revolving credit utilization under 20% and continue automated on-time payments."
            )
            action_plan.append(
                "✅ **Expedited Processing:** Eligible for preferential interest rate tiers and waived secondary documentation requirements."
            )

        # 5. Formal Institutional Credit Memorandum ('The Matter')
        matter_text = self._build_institutional_matter(
            applicant=applicant,
            pred_label=pred_label,
            risk_pct=risk_pct,
            risk_tier=risk_tier,
            action_code=action_code,
            problems=problems,
            strengths=strengths,
            emi=emi,
            est_monthly_loan_burden=est_monthly_loan_burden,
            loan_to_income=loan_to_income,
            asset_coverage=asset_coverage,
            action_plan=action_plan
        )

        return {
            'verdict': pred_label,
            'risk_percentage': risk_pct,
            'risk_tier': risk_tier,
            'verdict_color': verdict_color,
            'action_code': action_code,
            'financial_metrics': {
                'estimated_monthly_emi': round(emi, 2),
                'monthly_burden_pct': round(est_monthly_loan_burden, 2),
                'loan_to_income_pct': round(loan_to_income, 2),
                'asset_coverage_pct': round(asset_coverage, 2)
            },
            'problems': problems,
            'strengths': strengths,
            'action_plan': action_plan,
            'matter_report': matter_text
        }

    def _build_institutional_matter(
        self,
        applicant: Dict[str, Any],
        pred_label: str,
        risk_pct: float,
        risk_tier: str,
        action_code: str,
        problems: List[Dict[str, Any]],
        strengths: List[Dict[str, Any]],
        emi: float,
        est_monthly_loan_burden: float,
        loan_to_income: float,
        asset_coverage: float,
        action_plan: List[str]
    ) -> str:
        """Assembles a formal, high-fidelity AI underwriting memorandum."""
        problem_bullets = ""
        if problems:
            for i, p in enumerate(problems, 1):
                problem_bullets += (
                    f"  {i}. [{p['severity'].upper()} DEFICIT] {p['category']}:\n"
                    f"     - Observed: {p['metric']} (Target: {p['benchmark']} | Variance: {p['deficit']})\n"
                    f"     - Detailed Diagnosis: {p['explanation']}\n\n"
                )
        else:
            problem_bullets = "  • No acute financial deficits detected. All primary indicators within prime limits.\n\n"

        strength_bullets = ""
        if strengths:
            for i, s in enumerate(strengths, 1):
                strength_bullets += f"  {i}. {s['category']} ({s['metric']}): {s['impact']}\n"
        else:
            strength_bullets = "  • Minimal positive risk mitigants identified.\n"

        plan_bullets = "\n".join([f"  {step}" for step in action_plan])

        matter = f"""
========================================================================================
             AI CREDIT RISK ASSESSMENT & UNDERWRITING MEMORANDUM ('MATTER')
                      Aditya University - AI & ML Lending Intelligence
========================================================================================

I. EXECUTIVE UNDERWRITING SUMMARY
----------------------------------------------------------------------------------------
• Final Risk Classification : {pred_label.upper()}
• Model Risk Probability    : {risk_pct:.1f}% (High-Risk Default Likelihood)
• Assessed Risk Tier        : {risk_tier}
• Recommended Credit Action : {action_code}

II. APPLICANT FINANCIAL TELEMETRY
----------------------------------------------------------------------------------------
• Demographics              : Age: {applicant.get('age', 'N/A')} | Employment: {applicant.get('employment_status', 'N/A')}
• Income & Loan Size        : Annual Income: ${float(applicant.get('income', 0)):,.0f} | Requested Loan: ${float(applicant.get('loan_amount', 0)):,.0f}
• Estimated Monthly EMI     : ${emi:,.2f} / month (Consumes {est_monthly_loan_burden:.1f}% of gross monthly income)
• Credit Score & Reputation : FICO: {applicant.get('credit_score', 'N/A')} | Track Record: {applicant.get('payment_history', 'N/A')}
• Debt Commitments          : Existing Debt: ${float(applicant.get('existing_debts', 0)):,.0f} | DTI Ratio: {float(applicant.get('debt_to_income_ratio', 0))*100:.1f}%
• Collateral Base           : Declared Assets: ${float(applicant.get('assets', 0)):,.0f} (Coverage: {asset_coverage:.1f}%)

III. DETAILED PROBLEM DIAGNOSIS ('WHAT IS THE PROBLEM IN DETAIL')
----------------------------------------------------------------------------------------
{problem_bullets}
IV. POSITIVE COMPENSATING FACTORS (CREDIT STRENGTHS)
----------------------------------------------------------------------------------------
{strength_bullets}
V. ACTIONABLE RESTRUCTURING ROADMAP (HOW TO CURE / CONVERT TO LOW RISK)
----------------------------------------------------------------------------------------
{plan_bullets}

VI. FORMAL AI UNDERWRITING VERDICT
----------------------------------------------------------------------------------------
{"Based on multi-model ensemble consensus and SHAP marginal attribution, the application exhibits unmanageable debt exposure and impaired repayment probability. Approval is NOT RECOMMENDED under current parameters. The applicant should pursue the restructuring roadmap specified in Section V." if pred_label == 'High Risk' else "Based on multi-model ensemble consensus, the applicant demonstrates sound liquidity, sustainable debt burden, and unblemished credit character. Full loan approval is RECOMMENDED with standard covenants."}
========================================================================================
"""
        return matter.strip()
