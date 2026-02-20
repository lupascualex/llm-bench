"""Subprocess-based Python code execution for HumanEval."""

from __future__ import annotations

import asyncio


async def execute_python(
    code: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Execute Python code in a subprocess and return (success, output).

    Args:
        code: Complete Python code string to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        (True, stdout) if exit code 0, (False, stderr/error) otherwise.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"Execution timed out after {timeout}s"

        if proc.returncode == 0:
            return True, stdout.decode("utf-8", errors="replace")
        else:
            return False, stderr.decode("utf-8", errors="replace")

    except Exception as e:
        return False, str(e)


async def run_humaneval_test(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Run a HumanEval test case.

    Concatenates the function prompt, model completion, and test harness,
    then executes the combined code.
    """
    full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"
    return await execute_python(full_code, timeout=timeout)
