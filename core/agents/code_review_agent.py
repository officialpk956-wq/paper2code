from core.llm_client import llm_complete

def generate_code_review(
    problem_title: str,
    problem_description: str,
    test_cases: list,
    user_code: str,
    stdout: str,
    stderr: str,
    passed: bool,
    time_ms: int | None,
) -> str:
    """
    Call the LLM once to generate a code review.
    Returns a markdown string (200-400 words).
    """
    status = "PASSED" if passed else "FAILED"
    tc_summary = "\n".join(
        f"  Input: {tc.get('input', '')}  Expected: {tc.get('expected_output', '')}"
        for tc in (test_cases or [])[:3]  # show at most 3 test cases
    )
    prompt = f"""You are a senior software engineer reviewing a student's coding solution.

Problem: <problem_title>{problem_title}</problem_title>
Description: <problem_description>{problem_description[:800]}</problem_description>

Test cases (first 3):
<test_cases>{tc_summary}</test_cases>

Student's code:
<student_code>
{user_code[:2000]}
</student_code>
Execution result: {status}
stdout: {stdout[:500] if stdout else "(empty)"}
stderr: {stderr[:500] if stderr else "(empty)"}
Time: {time_ms}ms

Write a concise code review (200-400 words) in markdown:
{"- Explain exactly WHY the code failed and give ONE targeted hint (not the solution)" if not passed else "- Confirm what worked well, then suggest one optimization or edge case to consider"}

Be specific about line numbers or variable names when possible
Do NOT rewrite the solution for the student
Be encouraging but honest
"""
    return llm_complete(prompt)
