def generate_explanation(user_data, policy):
    """Generate detailed explanation for recommended policy"""
    
    # Format coverage items with proper capitalization and spacing
    coverage_items = [item.replace('_', ' ').title() for item in policy['coverage']]
    coverage_text = ', '.join(coverage_items)
    
    # Create main policy details section
    explanation = [
        f"Policy #{policy['policy_id']}",
        "\n📊 Basic Details:",
        f"• Premium: ${policy['premium_amount']:.2f} per month",
        f"• Waiting Period: {policy['waiting_period']} days",
        f"• Insurer Rating: {policy['insurer_rating']:.2f}/1.00",
        "\n🏥 Coverage Includes:",
        f"{coverage_text}"
    ]
    
    # Add personalized explanations in a separate section
    benefits = []
    
    if policy['premium_amount'] <= user_data['budget_amount']:
        benefits.append(f"\n• Within your budget of ${user_data['budget_amount']:.2f}")
    
    if user_data['maternity_coverage'] and 'maternity' in policy['coverage']:
        benefits.append("\n• Includes maternity coverage as requested")
    
    if user_data['insurer_rating_preference'] and policy['insurer_rating'] > 0.85:
        benefits.append(f"\n• High-rated insurer ({policy['insurer_rating']:.2f}/1.00)")
    
    if policy['waiting_period'] == 0:
        benefits.append("\n• No waiting period")
    elif policy['waiting_period'] <= 30:
        benefits.append(f"\n• Short waiting period of {policy['waiting_period']} days")
    
    coverage_count = len(policy['coverage'])
    if coverage_count >= 6:
        benefits.append(f"\n• Comprehensive coverage with {coverage_count} benefits")
    elif coverage_count >= 4:
        benefits.append(f"\n• Good coverage breadth with {coverage_count} benefits")
    
    if benefits:
        explanation.append("\n✨ Key Benefits:")
        explanation.extend(benefits)
    
    return "\n".join(explanation) 